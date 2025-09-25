import cv2
import numpy as np
import json
import re
from PIL import Image, ImageEnhance, ImageFilter
import moondream as md
import csv
import fitz

boxes_json_path = "template_boxes_3aths_v1.json" # created in template.py
template_pdf_path = "template_3aths_v1.pdf" # blank template pdf
photo_input_path = "photo.png" # photo of filled form
api_key = "your-api-key-here" # moondream api key

# align photo to the template
def align_form_using_logo(template_img, photo_path):
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    photo_img = cv2.imread(photo_path)
    # resize to roughly template size
    photo_img = cv2.resize(photo_img, (template_img.shape[1], template_img.shape[0]))
    photo_gray = cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(photo_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        print("Warning: not enough good matches for reliable alignment")
        return photo_img

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    height, width = template_img.shape[:2]
    aligned = cv2.warpPerspective(photo_img, H, (width, height))

    return aligned
  

# function to run moondream on each box from template
def extract_fields_from_aligned_image_moondream(image, boxes_json_path, api_key):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Init Moondream once
    model = md.vl(api_key=api_key)

    # folder to store images of boxes
     os.makedirs("rois", exist_ok=True)

    with open(boxes_json_path, "r") as f:
        box_coords = json.load(f)

    results = {}
    for label, (x1, y1, x2, y2) in box_coords.items():
        #roi = gray[y1:y2, x1:x2]

        # adding padding to boxes
        padding = 4
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(gray.shape[1], x2 + padding)
        y2_pad = min(gray.shape[0], y2 + padding)
        roi = gray[y1_pad:y2_pad, x1_pad:x2_pad]

        roi_pil = Image.fromarray(roi)

        # save boxes for inspection/debugging
        roi_pil.save(f"rois/debug_roi_{label}_moondream.png")

        # Query Moondream
        response = model.query(roi_pil, "Extract all handwritten text from inside this box. If blank, return an empty string. Expect mostly numbers or symbols unless clearly a word.")
        answer = response.get("answer", "").strip()

        results[label] = answer

    return results

doc = fitz.open(template_pdf_path)
pix = doc[0].get_pixmap()
template_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
template_img = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2BGR)

aligned_image = align_form_using_logo(template_img, photo_input_path)
cv2.imwrite("aligned_debug_output.png", aligned_image)
ocr_results_moondream = extract_fields_from_aligned_image_moondream(aligned_image, boxes_json_path, api_key)

print("Results:")
for label, value in ocr_results_moondream.items():
    print(f"{label}: {value}")


### CREATING TABLE & CSV FROM EXTRACTED DATA

with open("template_boxes_3aths_v1.json", "r") as f:
    cell_coords = json.load(f)

cell_values = ocr_results_moondream

# Combine coordinates + values
cells = []
for cell_id, coords in cell_coords.items():
    x1, y1, x2, y2 = coords
    value = cell_values.get(cell_id, "")

    # remove erroneous cell values
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

# Sort into rows
row_tolerance = 10  # tolerance for considering cells on the same row
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

# Add last row
if current_row:
    current_row.sort(key=lambda c: c["x"])
    rows.append(current_row)

# Write to csv
with open("output_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow([c["value"] for c in row])

print("CSV file created: output_table.csv")