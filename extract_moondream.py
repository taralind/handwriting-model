import cv2
import numpy as np
import json
import re
from PIL import Image, ImageEnhance, ImageFilter
import moondream as md
import csv
from dotenv import load_dotenv
import os

boxes_json_path = "templates_jsons/template_boxes_4aths_v1.json" # created in template.py
template_path = "templates/template_4aths_v1.png" # blank template png
photo_input_path = "photos/photo_4aths_v1.png" # png photo of filled form

# API KEY
load_dotenv()
api_key = os.getenv("MOONDREAM_API_KEY")

def align_form_using_logo(template_img, photo_path, use_sift=True):
    # Load photo
    photo_img = cv2.imread(photo_path)
    if photo_img is None:
        raise FileNotFoundError(f"Could not read image at {photo_path}")
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    photo_gray = cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)

    # Choose feature detector
    if use_sift and hasattr(cv2, 'SIFT_create'):
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
        flann_index = 1  # KDTREE
        index_params = dict(algorithm=flann_index, trees=5)
    else:
        detector = cv2.ORB_create(2000)
        norm_type = cv2.NORM_HAMMING
        flann_index = 6  # LSH
        index_params = dict(algorithm=flann_index,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)

    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(template_gray, None)
    kp2, des2 = detector.detectAndCompute(photo_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find features in one of the images.")

    # Match descriptors using KNN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe’s ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        raise ValueError("Not enough good matches to align images.")

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography could not be computed.")

    # Warp the photo to align with the template
    height, width = template_img.shape[:2]
    aligned = cv2.warpPerspective(photo_img, H, (width, height))

    return aligned

# function to run moondream on each box from template
def extract_fields_from_aligned_image_moondream(image, boxes_json_path, api_key):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Init Moondream once
    model = md.vl(api_key=api_key)

    with open(boxes_json_path, "r") as f:
        box_coords = json.load(f)

    results = {}
    for label, (x1, y1, x2, y2) in box_coords.items():

        # adding padding to boxes
        padding = 1
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(gray.shape[1], x2 + padding)
        y2_pad = min(gray.shape[0], y2 + padding)
        roi = gray[y1_pad:y2_pad, x1_pad:x2_pad]

        roi_pil = Image.fromarray(roi)

        # Query Moondream
        response = model.query(roi_pil, "Extract all handwritten text from inside this box. If blank, return an empty string. Expect mostly numbers or symbols unless clearly a word or letters. Ignore straight lines / box borders - these are not text. The box may contain two evenly spaced dots- if so, acknowledge these as decimal points between the numbers next to them.")
        answer = response.get("answer", "").strip()

        results[label] = answer

    return results

template_img = cv2.imread(template_path)
aligned_image = align_form_using_logo(template_img, photo_input_path)
cv2.imwrite("aligned_debug_output.png", aligned_image)
ocr_results_moondream = extract_fields_from_aligned_image_moondream(aligned_image, boxes_json_path, api_key)

print("Results:")
for label, value in ocr_results_moondream.items():
    print(f"{label}: {value}")


### CREATING TABLE & CSV FROM EXTRACTED DATA

def export_table_with_merged_cells(boxes_json_path, ocr_results, output_csv_path,
                                   row_tol=10, col_tol=10, size_threshold=0.5):
    """
    Rebuild a table from cell bounding boxes and OCR values, handling merged cells.

    Args:
        boxes_json_path: path to JSON file with box coordinates.
        ocr_results: dict mapping cell_id -> OCR text.
        output_csv_path: path to write CSV.
        row_tol: tolerance (pixels) to cluster rows.
        col_tol: tolerance (pixels) to cluster columns.
        size_threshold: ignore boxes that are bigger than fraction of table size.
    """

    # Load boxes
    with open(boxes_json_path, "r") as f:
        cell_coords = json.load(f)

    # Compute approximate table size
    all_boxes = np.array(list(cell_coords.values()))
    table_width = all_boxes[:, 2].max()
    table_height = all_boxes[:, 3].max()

    # Build cells with centers
    cells = []
    for cell_id, coords in cell_coords.items():
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1

        # Skip huge boxes that are likely the full table
        if w > size_threshold * table_width or h > size_threshold * table_height:
            continue

        value = ocr_results.get(cell_id, "").strip()
        cells.append({
            "id": cell_id,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": (x1 + x2)/2,
            "cy": (y1 + y2)/2,
            "value": value
        })

    if not cells:
        raise ValueError("No cells to process after filtering large boxes.")

    # --- Cluster row centers ---
    row_centers = sorted([c["cy"] for c in cells])
    rows = []
    current_row = [row_centers[0]]
    for cy in row_centers[1:]:
        if abs(cy - np.mean(current_row)) <= row_tol:
            current_row.append(cy)
        else:
            rows.append(np.mean(current_row))
            current_row = [cy]
    rows.append(np.mean(current_row))  # last row

    # --- Cluster column centers ---
    col_centers = sorted([c["cx"] for c in cells])
    cols = []
    current_col = [col_centers[0]]
    for cx in col_centers[1:]:
        if abs(cx - np.mean(current_col)) <= col_tol:
            current_col.append(cx)
        else:
            cols.append(np.mean(current_col))
            current_col = [cx]
    cols.append(np.mean(current_col))  # last column

    n_rows = len(rows)
    n_cols = len(cols)

    # --- Create empty grid ---
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    # --- Place cells in grid ---
    for c in cells:
        # Find rows the cell spans
        row_indices = [i for i, ry in enumerate(rows) if (c["y1"] <= ry <= c["y2"])]
        # Find columns the cell spans
        col_indices = [j for j, cx in enumerate(cols) if (c["x1"] <= cx <= c["x2"])]

        # Fill only the top-left slot of the spanned area
        if row_indices and col_indices:
            top_row = row_indices[0]
            left_col = col_indices[0]
            grid[top_row][left_col] = c["value"]

    # --- Write to CSV ---
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in grid:
            writer.writerow(row)

    print(f"Exported table to {output_csv_path}")
            
export_table_with_merged_cells(boxes_json_path, ocr_results_moondream, "output_table.csv")