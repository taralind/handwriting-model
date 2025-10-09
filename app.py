import streamlit as st
import cv2
import numpy as np
import json
import csv
from PIL import Image
import moondream as md
from dotenv import load_dotenv
import os
import io
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="Handwritten Form Processor", layout="wide")

# --- LOAD ENV + API KEY ---
load_dotenv()
api_key = os.getenv("MOONDREAM_API_KEY")

# --- FILE PATHS ---
# commented out while we've got template selection in the UI
#TEMPLATE_PATH = "templates/template_3aths_v1.png"
#BOXES_JSON_PATH = "templates_jsons/template_boxes_3aths_v1.json"

# --- FUNCTIONS ---
def align_form_using_logo(template_img, photo_array):
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    photo_gray = cv2.cvtColor(photo_array, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(photo_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    N_MATCHES = 300
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:N_MATCHES]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:N_MATCHES]]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    height, width = template_img.shape[:2]
    aligned = cv2.warpPerspective(photo_array, H, (width, height))
    return aligned


def extract_fields_from_aligned_image_moondream(image, boxes_json_path, api_key):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    model = md.vl(api_key=api_key)

    with open(boxes_json_path, "r") as f:
        box_coords = json.load(f)

    results = {}
    total_boxes = len(box_coords)

    # Streamlit progress bar setup
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, (label, (x1, y1, x2, y2)) in enumerate(box_coords.items(), start=1):
        padding = 1
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(gray.shape[1], x2 + padding)
        y2_pad = min(gray.shape[0], y2 + padding)
        roi = gray[y1_pad:y2_pad, x1_pad:x2_pad]
        roi_pil = Image.fromarray(roi)

        response = model.query(
            roi_pil,
            "Extract all handwritten text from inside this box. If blank, return an empty string. "
            "Expect mostly numbers or symbols unless clearly a word. Ignore straight lines / box borders - these are not text."
        )
        results[label] = response.get("answer", "").strip()

        # update progress bar
        progress = int((i / total_boxes) * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Extracting field {i}/{total_boxes}")

    progress_bar.empty()
    progress_text.empty()
    return results

def create_table_from_results(ocr_results, boxes_json_path):
    with open(boxes_json_path, "r") as f:
        cell_coords = json.load(f)

    cell_values = ocr_results
    cells = []

    for cell_id, coords in cell_coords.items():
        x1, y1, x2, y2 = coords
        value = cell_values.get(cell_id, "")
        if len(value) > 20:
            value = ""
        cells.append({
            "id": cell_id,
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "value": value
        })

    row_tolerance = 10
    cells.sort(key=lambda c: (c["y"], c["x"]))

    rows = []
    current_row = []
    last_y = None

    for cell in cells:
        if last_y is None or abs(cell["y"] - last_y) <= row_tolerance:
            current_row.append(cell)
        else:
            current_row.sort(key=lambda c: c["x"])
            rows.append(current_row)
            current_row = [cell]
        last_y = cell["y"]

    if current_row:
        current_row.sort(key=lambda c: c["x"])
        rows.append(current_row)

    data = [[c["value"] for c in row] for row in rows]
    df = pd.DataFrame(data)
    return df

# added so that the image alignment step is cached
# so that the extraction step doesn't re-run after pressing download to csv
@st.cache_data(show_spinner=False)
def get_ocr_results(aligned_img_bytes, boxes_json_path, api_key):
    import numpy as np
    import cv2
    nparr = np.frombuffer(aligned_img_bytes, np.uint8)
    aligned_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return extract_fields_from_aligned_image_moondream(aligned_img, boxes_json_path, api_key)


# --- STREAMLIT UI ---
st.title("Handwritten Form Detection")
#st.write("Description")

# --- TEMPLATE SELECTION ---
templates_folder = "templates"
boxes_folder = "templates_jsons"

# List available template PNGs
template_files = [f for f in os.listdir(templates_folder) if f.endswith(".png")]
template_files.sort()  # optional, sort alphabetically

# Ask user which template they used
selected_template_file = st.selectbox("Which template did you use?", template_files)

# Automatically find the corresponding JSON
# Assumes naming convention: template_XYZ.png -> template_boxes_XYZ.json
template_name_part = selected_template_file.replace("template_", "").replace(".png", "")
boxes_json_file = f"template_boxes_{template_name_part}.json"

# Full paths
TEMPLATE_PATH = os.path.join(templates_folder, selected_template_file)
BOXES_JSON_PATH = os.path.join(boxes_folder, boxes_json_file)

uploaded_file = st.file_uploader("Upload a fill-out form photo (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    photo_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    template_img = cv2.imread(TEMPLATE_PATH)

    with st.spinner("Aligning form to template..."):
        aligned_img = align_form_using_logo(template_img, photo_img)

    st.image(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB), caption="Aligned Image", width=400)

    # with st.spinner("Extracting text using Moondream..."):
    #     ocr_results = extract_fields_from_aligned_image_moondream(aligned_img, BOXES_JSON_PATH, api_key)
    # Convert aligned image to bytes for caching
    _, buf = cv2.imencode(".png", aligned_img)
    aligned_img_bytes = buf.tobytes()
    # Call the cached version
    with st.spinner("Extracting text using Moondream..."):
        ocr_results = get_ocr_results(aligned_img_bytes, BOXES_JSON_PATH, api_key)

    with st.spinner("Building output table..."):
        output_df = create_table_from_results(ocr_results, BOXES_JSON_PATH)

    st.success("Extraction complete!")
    st.dataframe(output_df)

    # CSV download
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download table as CSV",
        data=csv_buffer.getvalue(),
        file_name="output_table.csv",
        mime="text/csv"
    )