import cv2
import json
import numpy as np
from pdf2image import convert_from_path

# UPLOAD TEMPLATE
pdf_path = "template_3aths_v1.pdf"
output_json = "template_boxes_3aths_v1.json"

# Convert pdf to image
images = convert_from_path(pdf_path, dpi=300)
img = images[0]
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
img_copy = img.copy()

# Preprocess image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)  # invert for line detection
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 15, -2)

### Detect horizontal and vertical lines
horizontal = thresh.copy()
vertical = thresh.copy()
scale = 20 

# Horizontal lines
h_size = int(horizontal.shape[1] / scale)
h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
horizontal = cv2.erode(horizontal, h_structure)
horizontal = cv2.dilate(horizontal, h_structure)

# Vertical lines
v_size = int(vertical.shape[0] / scale)
v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
vertical = cv2.erode(vertical, v_structure)
vertical = cv2.dilate(vertical, v_structure)

# Combine lines
mask = cv2.add(horizontal, vertical)

# Find contours of each cell
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = {}
box_id = 1

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:  # filter tiny boxes / noise
        label = f"cell_{box_id}"
        boxes[label] = [int(x), int(y), int(x + w), int(y + h)]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)
        box_id += 1

# sort by top-left position
boxes = dict(sorted(boxes.items(), key=lambda b: (b[1][1], b[1][0])))

# Display detected cells for inspection
cv2.imshow("Detected Cells", img_copy)
print(f"Detected {len(boxes)} cells. Press any key to close the preview")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save json
with open(output_json, "w") as f:
    json.dump(boxes, f, indent=2)

print(f"\nSaved {len(boxes)} cells to {output_json}")
