#!/usr/bin/env python3
"""Test TensorRT layout detection (FP16 & FP8) with visual output.

Runs PP-DocLayoutV3 through the PaddleX tensorrt engine on real document
images, draws bounding boxes with labels and confidence scores, and saves
annotated results as PNG files.  Also prints a structured summary of all
detected regions.

Usage (inside container):
    python3 test_trt_layout_viz.py            # FP16 (default)
    python3 test_trt_layout_viz.py --fp8      # FP8
    python3 test_trt_layout_viz.py --both     # Run both, side-by-side
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from paddlex import create_predictor

# ── PP-DocLayoutV3 label list (25 categories) ──────────────────────────────
LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]

# Distinct colors per category (BGR for OpenCV) — 25 colors
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
    (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
    (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192),
    (64, 64, 64),
]

# Test images (paths inside container)
IMAGE_DIR = Path("/tmp/test_images")
OUTPUT_DIR = Path("/tmp/test_output")


def get_test_images() -> list[Path]:
    """Return list of test images, sorted by name."""
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image dir not found: {IMAGE_DIR}")
    imgs = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not imgs:
        raise FileNotFoundError(f"No images in {IMAGE_DIR}")
    return imgs


def draw_boxes(
    image: np.ndarray,
    boxes: list[dict],
    min_score: float = 0.25,
) -> np.ndarray:
    """Draw bounding boxes with labels and scores on the image.

    Args:
        image: BGR image (H, W, 3) uint8
        boxes: list of box dicts with 'label', 'score', 'coordinate'
        min_score: minimum confidence to draw

    Returns:
        Annotated copy of the image
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    # Scale font/thickness based on image size
    base_size = min(img_h, img_w)
    font_scale = max(0.4, base_size / 1200)
    thickness = max(1, int(base_size / 600))
    box_thickness = max(2, int(base_size / 400))

    for box in boxes:
        score = box.get("score", 0.0)
        if score < min_score:
            continue

        label = box.get("label", "unknown")
        cls_id = box.get("cls_id", 0)
        coord = box.get("coordinate", [0, 0, 0, 0])
        order = box.get("order", 0) or 0

        x1, y1, x2, y2 = [int(v) for v in coord]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        color = COLORS[cls_id % len(COLORS)]

        # Draw filled semi-transparent box
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # Draw solid border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Label text with background
        text = f"{label} {score:.2f}"
        if order:
            text = f"#{order} {text}"

        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness)
        )
        # Position label above the box (or inside if too close to top)
        ly = y1 - th - baseline - 4 if y1 > th + baseline + 10 else y1 + 4
        lx = x1

        cv2.rectangle(
            annotated,
            (lx, ly - 2),
            (lx + tw + 6, ly + th + baseline + 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (lx + 3, ly + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness),
            cv2.LINE_AA,
        )

    return annotated


def draw_legend(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Draw a legend in the top-right corner showing detected categories."""
    # Get unique labels actually detected
    seen = {}
    for box in boxes:
        label = box.get("label", "unknown")
        cls_id = box.get("cls_id", 0)
        if label not in seen:
            seen[label] = cls_id

    if not seen:
        return image

    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    font_scale = max(0.35, min(img_h, img_w) / 1500)
    thickness = 1
    line_h = int(20 * font_scale * 2.5)
    box_w = int(180 * font_scale * 3)

    # Legend background
    n_lines = len(seen) + 1  # +1 for title
    legend_h = n_lines * line_h + 10
    legend_x = img_w - box_w - 10
    legend_y = 10

    cv2.rectangle(
        annotated,
        (legend_x, legend_y),
        (legend_x + box_w, legend_y + legend_h),
        (30, 30, 30),
        -1,
    )
    cv2.rectangle(
        annotated,
        (legend_x, legend_y),
        (legend_x + box_w, legend_y + legend_h),
        (200, 200, 200),
        1,
    )

    # Title
    cv2.putText(
        annotated,
        "Detected Regions",
        (legend_x + 8, legend_y + int(line_h * 0.8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 1.1,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Color swatches + labels
    for i, (label, cls_id) in enumerate(sorted(seen.items())):
        y = legend_y + (i + 1) * line_h + int(line_h * 0.6)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(
            annotated,
            (legend_x + 8, y - int(line_h * 0.4)),
            (legend_x + 8 + int(15 * font_scale * 2),
             y + int(line_h * 0.4)),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (legend_x + 8 + int(20 * font_scale * 2) + 5, y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def run_precision(
    precision: str,
    images: list[Path],
    n_calib: int = 10,
) -> list[dict]:
    """Run inference at a given precision on all images.

    Returns list of result dicts with image path, boxes, timing, annotated path.
    """
    print(f"\n{'='*70}")
    print(f"  Running PP-DocLayoutV3 with precision={precision}")
    print(f"{'='*70}")

    # Create predictor
    engine_config = {
        "precision": precision,
        "device_id": 0,
        "model_type": "layout",
        "n_calib": n_calib,
    }

    print(f"\n  Creating predictor (precision={precision})...")
    t0 = time.time()
    predictor = create_predictor(
        "PP-DocLayoutV3",
        engine="tensorrt",
        engine_config=engine_config,
    )
    create_time = time.time() - t0
    print(f"  Predictor ready in {create_time:.1f}s")

    # Output subdirectory
    out_sub = OUTPUT_DIR / precision
    out_sub.mkdir(parents=True, exist_ok=True)

    results = []

    # Warmup: run a dummy inference first
    print("\n  Warmup inference...")
    dummy = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    predictor.predict(dummy)

    for img_path in images:
        print(f"\n  Processing: {img_path.name}")

        # Load image (PaddleX expects BGR via cv2)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ERROR: could not read {img_path}")
            continue
        print(f"    Image: {img.shape[1]}x{img.shape[0]} ({img.shape})")

        # Run inference
        t0 = time.time()
        result_gen = predictor.predict(img)
        # Consume the generator
        result_list = list(result_gen)
        elapsed = time.time() - t0
        print(f"    Inference: {elapsed:.3f}s")

        if not result_list:
            print("    ERROR: no results returned")
            continue

        result = result_list[0]
        boxes = result.get("boxes", []) if isinstance(result, dict) else []
        # Also try attribute access
        if not boxes and hasattr(result, "get"):
            boxes = result.get("boxes", [])
        if not boxes and hasattr(result, "boxes"):
            boxes = result.boxes
        if not boxes and isinstance(result, dict):
            boxes = result.get("boxes", [])

        print(f"    Detected {len(boxes)} regions")

        # Print box details
        for i, box in enumerate(boxes):
            label = box.get("label", "?")
            score = box.get("score", 0.0)
            coord = box.get("coordinate", [0, 0, 0, 0])
            order = box.get("order", 0) or 0
            print(
                f"      [{i:2d}] #{order:2d} {label:20s} "
                f"score={score:.3f}  bbox=({coord[0]:.0f},{coord[1]:.0f})"
                f"-({coord[2]:.0f},{coord[3]:.0f})"
            )

        # Draw annotated image
        annotated = draw_boxes(img, boxes, min_score=0.25)
        annotated = draw_legend(annotated, boxes)

        # Save
        out_name = f"{img_path.stem}_{precision}.png"
        out_path = out_sub / out_name
        cv2.imwrite(str(out_path), annotated)
        print(f"    Saved: {out_path}")

        # Also save raw detection JSON
        json_name = f"{img_path.stem}_{precision}.json"
        json_path = out_sub / json_name
        box_data = []
        for box in boxes:
            box_data.append({
                "label": box.get("label", ""),
                "cls_id": box.get("cls_id", -1),
                "score": float(box.get("score", 0.0)),
                "coordinate": [float(v) for v in box.get("coordinate", [])],
                "order": box.get("order", 0) or 0,
            })
        with open(json_path, "w") as f:
            json.dump({
                "image": img_path.name,
                "image_size": [img.shape[1], img.shape[0]],
                "precision": precision,
                "inference_time_s": elapsed,
                "num_detections": len(boxes),
                "boxes": box_data,
            }, f, indent=2)
        print(f"    JSON:  {json_path}")

        results.append({
            "image": img_path.name,
            "precision": precision,
            "inference_time_s": elapsed,
            "num_detections": len(boxes),
            "boxes": box_data,
            "annotated_path": str(out_path),
        })

    return results


def compare_results(fp16_results, fp8_results):
    """Print a comparison table of FP16 vs FP8 results."""
    print(f"\n{'='*70}")
    print("  COMPARISON: FP16 vs FP8")
    print(f"{'='*70}")
    print(f"  {'Image':<35s} {'FP16 det':>10s} {'FP8 det':>10s} "
          f"{'FP16 ms':>10s} {'FP8 ms':>10s} {'Δ det':>6s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    for r16, r8 in zip(fp16_results, fp8_results):
        name = r16["image"][:35]
        n16 = r16["num_detections"]
        n8 = r8["num_detections"]
        t16 = r16["inference_time_s"] * 1000
        t8 = r8["inference_time_s"] * 1000
        delta = n8 - n16
        print(f"  {name:<35s} {n16:>10d} {n8:>10d} "
              f"{t16:>10.1f} {t8:>10.1f} {delta:>+6d}")

    print(f"{'='*70}")


def main():
    ap = argparse.ArgumentParser(
        description="Test TensorRT layout detection with visual output"
    )
    ap.add_argument(
        "--fp8", action="store_true",
        help="Run FP8 precision (default: FP16)",
    )
    ap.add_argument(
        "--both", action="store_true",
        help="Run both FP16 and FP8, then compare",
    )
    ap.add_argument(
        "--n-calib", type=int, default=10,
        help="Number of calibration images for FP8 (default: 10)",
    )
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = get_test_images()
    print(f"\nTest images ({len(images)}):")
    for img in images:
        print(f"  - {img.name}")

    if args.both:
        fp16_results = run_precision("fp16", images, args.n_calib)
        fp8_results = run_precision("fp8", images, args.n_calib)
        compare_results(fp16_results, fp8_results)

        # Save comparison summary
        summary_path = OUTPUT_DIR / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "fp16": fp16_results,
                "fp8": fp8_results,
            }, f, indent=2)
        print(f"\nComparison summary saved to: {summary_path}")

    elif args.fp8:
        run_precision("fp8", images, args.n_calib)

    else:
        run_precision("fp16", images, args.n_calib)

    print(f"\nDone! Annotated images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
