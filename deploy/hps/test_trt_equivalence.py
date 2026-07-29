#!/usr/bin/env python3
"""Functional equivalence test: TensorRT engine vs original Paddle engine.

Runs PP-DocLayoutV3 through both the original Paddle inference engine and
the TensorRT engine (FP16 + FP8), then compares detection results to
verify that the TRT path produces functionally equivalent output.

Compares:
  - Number of detected boxes
  - Bounding box coordinates (IoU + pixel delta)
  - Class labels (exact match)
  - Confidence scores (delta)
  - Reading order

Usage (inside container):
    python3 test_trt_equivalence.py
    python3 test_trt_equivalence.py --fp8
    python3 test_trt_equivalence.py --both
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from paddlex import create_predictor

IMAGE_DIR = Path("/tmp/test_images")
OUTPUT_DIR = Path("/tmp/test_output")


def get_test_images() -> list[Path]:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image dir not found: {IMAGE_DIR}")
    imgs = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not imgs:
        raise FileNotFoundError(f"No images in {IMAGE_DIR}")
    return imgs


def extract_boxes(result) -> list[dict]:
    """Extract normalized box list from a predictor result."""
    if isinstance(result, dict):
        boxes = result.get("boxes", [])
    elif hasattr(result, "boxes"):
        boxes = result.boxes
    else:
        boxes = []
    # Normalize: ensure each box has all fields
    out = []
    for b in boxes:
        out.append({
            "label": b.get("label", ""),
            "cls_id": b.get("cls_id", -1),
            "score": float(b.get("score", 0.0)),
            "coordinate": [float(v) for v in b.get("coordinate", [0, 0, 0, 0])],
            "order": b.get("order", 0) or 0,
        })
    return out


def iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    ref_boxes: list[dict],
    cmp_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Match boxes between two sets greedily by IoU.

    Returns list of match dicts:
      {"ref_idx", "cmp_idx", "iou", "label_match", "score_delta",
       "coord_delta", "ref": {...}, "cmp": {...}}
    Unmatched boxes get idx=-1.
    """
    matches = []
    used_cmp = set()

    for i, rb in enumerate(ref_boxes):
        best_j = -1
        best_iou = 0.0
        for j, cb in enumerate(cmp_boxes):
            if j in used_cmp:
                continue
            # Prefer same label, but also consider cross-label
            i_val = iou(rb["coordinate"], cb["coordinate"])
            if i_val > best_iou:
                best_iou = i_val
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold:
            used_cmp.add(best_j)
            cb = cmp_boxes[best_j]
            coord_delta = max(
                abs(a - b) for a, b in zip(rb["coordinate"], cb["coordinate"])
            )
            matches.append({
                "ref_idx": i,
                "cmp_idx": best_j,
                "iou": round(best_iou, 4),
                "label_match": rb["label"] == cb["label"],
                "cls_match": rb["cls_id"] == cb["cls_id"],
                "score_delta": round(cb["score"] - rb["score"], 6),
                "coord_delta_px": round(coord_delta, 4),
                "order_match": rb["order"] == cb["order"],
                "ref": rb,
                "cmp": cb,
            })
        else:
            matches.append({
                "ref_idx": i,
                "cmp_idx": -1,
                "iou": 0.0,
                "label_match": False,
                "ref": rb,
                "cmp": None,
                "note": "unmatched in ref",
            })

    # Add unmatched cmp boxes
    for j, cb in enumerate(cmp_boxes):
        if j not in used_cmp:
            matches.append({
                "ref_idx": -1,
                "cmp_idx": j,
                "iou": 0.0,
                "label_match": False,
                "ref": None,
                "cmp": cb,
                "note": "unmatched in cmp",
            })

    return matches


def run_engine(
    engine_name: str,
    images: list[Path],
    precision: str | None = None,
    n_calib: int = 10,
) -> dict[str, list[dict]]:
    """Run inference with a given engine on all images.

    Returns dict mapping image_name -> list of box dicts.
    """
    print(f"\n{'='*70}")
    if engine_name == "tensorrt":
        print(f"  Engine: TensorRT (precision={precision})")
    else:
        print(f"  Engine: {engine_name} (original Paddle)")
    print(f"{'='*70}")

    if engine_name == "tensorrt":
        engine_config = {
            "precision": precision,
            "device_id": 0,
            "model_type": "layout",
            "n_calib": n_calib,
        }
        print(f"  Creating predictor...")
        t0 = time.time()
        predictor = create_predictor(
            "PP-DocLayoutV3",
            engine="tensorrt",
            engine_config=engine_config,
        )
    else:
        print(f"  Creating predictor (original paddle engine)...")
        t0 = time.time()
        predictor = create_predictor(
            "PP-DocLayoutV3",
            engine=engine_name,
            device="gpu",
        )
    print(f"  Predictor ready in {time.time() - t0:.1f}s")

    # Warmup
    print("  Warmup...")
    dummy = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    list(predictor.predict(dummy))

    results = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ERROR: could not read {img_path}")
            continue

        t0 = time.perf_counter()
        result_list = list(predictor.predict(img))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = extract_boxes(result_list[0] if result_list else {})

        tag = precision if engine_name == "tensorrt" else engine_name
        print(f"  {img_path.name}: {len(boxes)} boxes in {elapsed_ms:.1f}ms")
        results[img_path.name] = boxes

    return results


def compare_engines(
    ref_name: str,
    ref_results: dict[str, list[dict]],
    cmp_name: str,
    cmp_results: dict[str, list[dict]],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Compare two engines' results. ref is the baseline."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {ref_name} (baseline) vs {cmp_name}")
    print(f"{'='*70}")

    all_comparisons = []

    for img_name in sorted(ref_results.keys()):
        ref_boxes = ref_results[img_name]
        cmp_boxes = cmp_results.get(img_name, [])

        matches = match_boxes(ref_boxes, cmp_boxes, iou_threshold)

        # Stats
        n_ref = len(ref_boxes)
        n_cmp = len(cmp_boxes)
        matched = [m for m in matches if m["ref_idx"] >= 0 and m["cmp_idx"] >= 0]
        unmatched_ref = [m for m in matches if m["cmp_idx"] == -1]
        unmatched_cmp = [m for m in matches if m["ref_idx"] == -1]
        label_mismatches = [m for m in matched if not m["label_match"]]

        ious = [m["iou"] for m in matched]
        coord_deltas = [m["coord_delta_px"] for m in matched]
        score_deltas = [abs(m["score_delta"]) for m in matched]

        mean_iou = sum(ious) / len(ious) if ious else 0.0
        max_coord = max(coord_deltas) if coord_deltas else 0.0
        mean_score_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
        max_score_delta = max(score_deltas) if score_deltas else 0.0

        status = "EQUIVALENT" if (
            len(unmatched_ref) == 0 and
            len(unmatched_cmp) == 0 and
            len(label_mismatches) == 0 and
            max_coord < 5.0  # within 5px
        ) else "DIFFERS"

        print(f"\n  {img_name}:")
        print(f"    {ref_name}: {n_ref} boxes | {cmp_name}: {n_cmp} boxes")
        print(f"    Matched: {len(matched)}  Unmatched ref: {len(unmatched_ref)}  "
              f"Unmatched cmp: {len(unmatched_cmp)}")
        print(f"    Label mismatches: {len(label_mismatches)}")
        print(f"    Mean IoU: {mean_iou:.4f}  Max coord delta: {max_coord:.2f}px")
        print(f"    Score delta: mean={mean_score_delta:.6f}  "
              f"max={max_score_delta:.6f}")
        print(f"    Status: {status}")

        if unmatched_ref:
            print(f"    -- Boxes in {ref_name} but not {cmp_name}:")
            for m in unmatched_ref:
                rb = m["ref"]
                print(f"       {rb['label']:20s} score={rb['score']:.3f} "
                      f"bbox=({rb['coordinate'][0]:.0f},{rb['coordinate'][1]:.0f})"
                      f"-({rb['coordinate'][2]:.0f},{rb['coordinate'][3]:.0f})")

        if unmatched_cmp:
            print(f"    -- Boxes in {cmp_name} but not {ref_name}:")
            for m in unmatched_cmp:
                cb = m["cmp"]
                print(f"       {cb['label']:20s} score={cb['score']:.3f} "
                      f"bbox=({cb['coordinate'][0]:.0f},{cb['coordinate'][1]:.0f})"
                      f"-({cb['coordinate'][2]:.0f},{cb['coordinate'][3]:.0f})")

        if label_mismatches:
            print(f"    -- Label mismatches:")
            for m in label_mismatches:
                print(f"       ref: {m['ref']['label']} ({m['ref']['score']:.3f})  "
                      f"cmp: {m['cmp']['label']} ({m['cmp']['score']:.3f})  "
                      f"IoU={m['iou']:.3f}")

        if max_coord >= 1.0 and matched:
            print(f"    -- Boxes with coordinate delta >= 1px:")
            for m in matched:
                if m["coord_delta_px"] >= 1.0:
                    print(f"       {m['ref']['label']:20s}  "
                          f"delta={m['coord_delta_px']:.2f}px  "
                          f"ref=({m['ref']['coordinate'][0]:.1f},"
                          f"{m['ref']['coordinate'][1]:.1f},"
                          f"{m['ref']['coordinate'][2]:.1f},"
                          f"{m['ref']['coordinate'][3]:.1f})  "
                          f"cmp=({m['cmp']['coordinate'][0]:.1f},"
                          f"{m['cmp']['coordinate'][1]:.1f},"
                          f"{m['cmp']['coordinate'][2]:.1f},"
                          f"{m['cmp']['coordinate'][3]:.1f})")

        all_comparisons.append({
            "image": img_name,
            "ref_engine": ref_name,
            "cmp_engine": cmp_name,
            "n_ref_boxes": n_ref,
            "n_cmp_boxes": n_cmp,
            "n_matched": len(matched),
            "n_unmatched_ref": len(unmatched_ref),
            "n_unmatched_cmp": len(unmatched_cmp),
            "n_label_mismatches": len(label_mismatches),
            "mean_iou": round(mean_iou, 4),
            "max_coord_delta_px": round(max_coord, 4),
            "mean_score_delta": round(mean_score_delta, 6),
            "max_score_delta": round(max_score_delta, 6),
            "status": status,
            "matches": [
                {
                    "ref_label": m["ref"]["label"] if m["ref"] else None,
                    "cmp_label": m["cmp"]["label"] if m["cmp"] else None,
                    "iou": m.get("iou", 0.0),
                    "label_match": m.get("label_match", False),
                    "score_delta": m.get("score_delta", 0.0),
                    "coord_delta_px": m.get("coord_delta_px", 0.0),
                }
                for m in matches
            ],
        })

    return all_comparisons


def main():
    ap = argparse.ArgumentParser(
        description="Functional equivalence test: TRT vs original Paddle engine"
    )
    ap.add_argument(
        "--fp8", action="store_true",
        help="Compare original vs TRT-FP8 (default: TRT-FP16)",
    )
    ap.add_argument(
        "--both", action="store_true",
        help="Compare original vs TRT-FP16 AND TRT-FP8",
    )
    ap.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for box matching (default: 0.5)",
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

    # Always run original Paddle engine first as baseline
    print("\n" + "="*70)
    print("  STEP 1: Run original Paddle engine (baseline)")
    print("="*70)
    paddle_results = run_engine("paddle", images)

    comparisons = []

    if args.both or not args.fp8:
        print("\n" + "="*70)
        print("  STEP 2: Run TensorRT FP16")
        print("="*70)
        trt_fp16_results = run_engine("tensorrt", images, precision="fp16", n_calib=args.n_calib)
        comp16 = compare_engines(
            "paddle (original)", paddle_results,
            "tensorrt-fp16", trt_fp16_results,
            args.iou_threshold,
        )
        comparisons.append({"comparison": "paddle vs tensorrt-fp16", "results": comp16})

    if args.both or args.fp8:
        print("\n" + "="*70)
        print("  STEP 2: Run TensorRT FP8")
        print("="*70)
        trt_fp8_results = run_engine("tensorrt", images, precision="fp8", n_calib=args.n_calib)
        comp8 = compare_engines(
            "paddle (original)", paddle_results,
            "tensorrt-fp8", trt_fp8_results,
            args.iou_threshold,
        )
        comparisons.append({"comparison": "paddle vs tensorrt-fp8", "results": comp8})

    # Save results
    out_path = OUTPUT_DIR / "equivalence_results.json"
    with open(out_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"\nEquivalence results saved to: {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    for comp in comparisons:
        name = comp["comparison"]
        results = comp["results"]
        n_equiv = sum(1 for r in results if r["status"] == "EQUIVALENT")
        n_total = len(results)
        print(f"  {name}: {n_equiv}/{n_total} images EQUIVALENT")
        for r in results:
            status_icon = "✓" if r["status"] == "EQUIVALENT" else "✗"
            print(f"    {status_icon} {r['image']:<40s}  "
                  f"ref={r['n_ref_boxes']}  cmp={r['n_cmp_boxes']}  "
                  f"matched={r['n_matched']}  "
                  f"IoU={r['mean_iou']:.3f}  "
                  f"maxΔcoord={r['max_coord_delta_px']:.1f}px  "
                  f"maxΔscore={r['max_score_delta']:.4f}")

    print(f"{'='*70}")
    print("  Legend: ✓ = functionally equivalent (same boxes, labels, coords within 5px)")
    print("          ✗ = differences detected (see details above)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
