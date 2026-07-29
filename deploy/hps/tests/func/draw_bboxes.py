#!/usr/bin/env python3
"""
Draw bounding boxes from API JSON output onto original images and save them.

Usage: python3 draw_bboxes.py
Reads images from /tmp/test_images/, sends each to the API, extracts bbox
info from the json_content response, draws colored boxes + labels, saves
annotated images to /tmp/bbox_output/.
"""

import json
import os
import time
import urllib.request
import urllib.parse
from PIL import Image, ImageDraw, ImageFont

API_URL = "http://127.0.0.1:8090/v1/convert/file"
INPUT_DIR = "/tmp/test_images"
OUTPUT_DIR = "/tmp/bbox_output"

# Color map for different element labels
LABEL_COLORS = {
    "title":            (255, 0, 0),      # red
    "section_header":   (255, 140, 0),    # dark orange
    "text":             (0, 128, 0),      # green
    "paragraph":        (0, 128, 0),      # green
    "list_item":        (0, 0, 255),      # blue
    "caption":          (128, 0, 128),    # purple
    "footnote":         (128, 128, 0),    # olive
    "formula":          (255, 0, 255),    # magenta
    "page_header":      (0, 128, 128),    # teal
    "page_footer":      (0, 128, 128),    # teal
    "picture":          (255, 165, 0),    # orange
    "table":            (0, 0, 128),      # navy
    "checkbox_selected": (255, 20, 147),  # deep pink
    "checkbox_unselected": (255, 105, 180),# hot pink
    "key_value_region": (75, 0, 130),     # indigo
}
DEFAULT_COLOR = (100, 100, 100)  # gray

def get_color(label):
    return LABEL_COLORS.get(label, DEFAULT_COLOR)

def try_font(size):
    """Try to find a usable font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def send_image(filepath):
    """Send image to API, return the parsed JSON response."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_data = f.read()

    # Build multipart form data
    boundary = "----draw_bbox_boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="image_density"\r\n\r\n'
        f"72\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    url = f"{API_URL}?to_formats=json"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())

def extract_elements(doc):
    """Extract all elements with bboxes from a DoclingDocument JSON.

    Returns list of (label, bbox, text_preview) tuples.
    bbox is (l, t, r, b) in page coordinates.
    """
    elements = []

    # Text elements
    for t in doc.get("texts", []):
        label = t.get("label", "text")
        text_preview = (t.get("text") or "")[:40]
        for prov in t.get("prov", []):
            bbox = prov.get("bbox")
            if bbox and all(k in bbox for k in ("l", "t", "r", "b")):
                elements.append((label, (bbox["l"], bbox["t"], bbox["r"], bbox["b"]), text_preview))

    # Pictures
    for p in doc.get("pictures", []):
        for prov in p.get("prov", []):
            bbox = prov.get("bbox")
            if bbox and all(k in bbox for k in ("l", "t", "r", "b")):
                elements.append(("picture", (bbox["l"], bbox["t"], bbox["r"], bbox["b"]), "[picture]"))

    # Tables
    for tbl in doc.get("tables", []):
        for prov in tbl.get("prov", []):
            bbox = prov.get("bbox")
            if bbox and all(k in bbox for k in ("l", "t", "r", "b")):
                elements.append(("table", (bbox["l"], bbox["t"], bbox["r"], bbox["b"]), "[table]"))

    return elements

def draw_boxes_on_image(image_path, elements, output_path):
    """Draw bounding boxes on the image and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Scale factor: page coords may differ from image pixel coords
    page_w, page_h = None, None
    # Try to get page size from the first element's coordinate range
    # Actually we need the page size from the doc — but we'll just
    # use the image size since page coords should match

    font = try_font(max(12, img.width // 80))
    label_font = try_font(max(10, img.width // 100))

    # Track label counts for summary
    label_counts = {}

    for label, (l, t, r, b), text_preview in elements:
        color = get_color(label)
        label_counts[label] = label_counts.get(label, 0) + 1

        # Draw rectangle (2px width)
        draw.rectangle([l, t, r, b], outline=color, width=2)

        # Draw label background + text above the box
        label_text = label
        if text_preview:
            label_text = f"{label}: {text_preview}"

        # Measure text
        try:
            bbox = draw.textbbox((0, 0), label_text, font=label_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(label_text) * 7, 12

        # Position label above the box (or below if near top)
        ly = t - th - 2 if t - th - 2 > 0 else t + 2
        lx = l

        draw.rectangle([lx, ly, lx + tw + 4, ly + th + 2], fill=color)
        draw.text((lx + 2, ly + 1), label_text, fill=(255, 255, 255), font=label_font)

    img.save(output_path)
    print(f"  Saved: {output_path} ({img.width}x{img.height}, {len(elements)} boxes)")
    print(f"  Label counts: {label_counts}")
    return label_counts

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print("No images found in", INPUT_DIR)
        return

    print(f"Found {len(images)} images to process\n")

    total_elements = 0
    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_bbox.jpg")

        print(f"Processing: {img_name}")
        print(f"  Sending to API...")

        try:
            resp = send_image(img_path)
        except Exception as e:
            print(f"  ERROR: API call failed: {e}")
            continue

        status = resp.get("status", "unknown")
        if status != "success":
            print(f"  ERROR: API returned status={status}")
            errors = resp.get("errors", [])
            if errors:
                for err in errors:
                    print(f"    {err.get('module_name', '?')}: {err.get('error_message', '?')}")
            continue

        doc = resp.get("document", {}).get("json_content")
        if doc is None:
            print(f"  ERROR: No json_content in response")
            continue

        elements = extract_elements(doc)
        print(f"  Extracted {len(elements)} elements with bounding boxes")

        counts = draw_boxes_on_image(img_path, elements, out_path)
        total_elements += len(elements)
        print()

    print(f"Done! Total {total_elements} bounding boxes drawn across {len(images)} images.")
    print(f"Output saved to {OUTPUT_DIR}/")
    print(f"\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")

if __name__ == "__main__":
    main()
