#!/usr/bin/env python3
"""Benchmark TensorRT layout detection latency variance.

Runs PP-DocLayoutV3 through the PaddleX tensorrt engine multiple times per
image to measure latency statistics (mean, std, min, max, p50, p95, p99)
and detection consistency (does the box count / coordinates vary across
runs?).

Usage (inside container):
    python3 test_trt_variance.py                 # FP16, 20 runs
    python3 test_trt_variance.py --fp8           # FP8
    python3 test_trt_variance.py --both          # Both precisions
    python3 test_trt_variance.py --runs 50       # 50 runs per image
"""
from __future__ import annotations

import argparse
import json
import time
import statistics
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


def percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile (0-100)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_variance_benchmark(
    precision: str,
    images: list[Path],
    n_runs: int = 20,
    n_warmup: int = 5,
    n_calib: int = 10,
) -> list[dict]:
    """Run n_runs inference passes per image and collect timing stats."""

    print(f"\n{'='*70}")
    print(f"  Variance Benchmark: precision={precision}, runs={n_runs}")
    print(f"{'='*70}")

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
    print(f"  Predictor ready in {time.time() - t0:.1f}s")

    # Warmup
    print(f"\n  Warmup ({n_warmup} runs)...")
    dummy = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    for _ in range(n_warmup):
        list(predictor.predict(dummy))

    all_results = []

    for img_path in images:
        print(f"\n  Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ERROR: could not read {img_path}")
            continue
        print(f"    Image: {img.shape[1]}x{img.shape[0]}")

        latencies_ms = []
        box_counts = []
        # Collect first-run boxes for consistency check
        first_boxes = None
        coord_variations = []  # track max coordinate delta across runs

        for run_idx in range(n_runs):
            t0 = time.perf_counter()
            result_list = list(predictor.predict(img))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            latencies_ms.append(elapsed_ms)

            result = result_list[0] if result_list else {}
            boxes = result.get("boxes", []) if isinstance(result, dict) else []
            box_counts.append(len(boxes))

            if run_idx == 0:
                first_boxes = boxes
            else:
                # Compare coordinates with first run
                if len(boxes) == len(first_boxes):
                    max_delta = 0.0
                    for b1, b2 in zip(first_boxes, boxes):
                        c1 = b1.get("coordinate", [0, 0, 0, 0])
                        c2 = b2.get("coordinate", [0, 0, 0, 0])
                        for v1, v2 in zip(c1, c2):
                            max_delta = max(max_delta, abs(v1 - v2))
                    coord_variations.append(max_delta)
                else:
                    coord_variations.append(-1)  # box count mismatch

        # Compute statistics
        mean_ms = statistics.mean(latencies_ms)
        std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        min_ms = min(latencies_ms)
        max_ms = max(latencies_ms)
        p50 = percentile(latencies_ms, 50)
        p95 = percentile(latencies_ms, 95)
        p99 = percentile(latencies_ms, 99)

        # Detection consistency
        unique_counts = set(box_counts)
        count_consistent = len(unique_counts) == 1

        coord_max_var = max(coord_variations) if coord_variations else 0.0
        coord_consistent = all(v == 0.0 for v in coord_variations) if coord_variations else True

        print(f"    Latency (ms): mean={mean_ms:.2f}  std={std_ms:.2f}  "
              f"min={min_ms:.2f}  max={max_ms:.2f}")
        print(f"    Percentiles:  p50={p50:.2f}  p95={p95:.2f}  p99={p99:.2f}")
        print(f"    Box count:     {box_counts[0]} (consistent={count_consistent})")
        if not count_consistent:
            print(f"      Counts across runs: {box_counts}")
        print(f"    Coord variance: max_delta={coord_max_var:.4f} px "
              f"(consistent={coord_consistent})")
        if not coord_consistent and coord_max_var > 0:
            print(f"      Max coord deltas per run: "
                  f"{[f'{v:.3f}' for v in coord_variations]}")

        # Also print score variance for first few boxes
        if first_boxes and n_runs > 1:
            # Re-run to collect scores
            all_scores_per_box = [[] for _ in range(len(first_boxes))]
            for _ in range(min(n_runs, 10)):
                rl = list(predictor.predict(img))
                r = rl[0] if rl else {}
                bs = r.get("boxes", []) if isinstance(r, dict) else []
                if len(bs) == len(first_boxes):
                    for i, b in enumerate(bs):
                        all_scores_per_box[i].append(b.get("score", 0.0))
            print(f"    Score variance (first 5 boxes):")
            for i, scores in enumerate(all_scores_per_box[:5]):
                if len(scores) > 1:
                    lbl = first_boxes[i].get("label", "?")
                    s_mean = statistics.mean(scores)
                    s_std = statistics.stdev(scores)
                    print(f"      [{i}] {lbl:20s}  mean={s_mean:.4f}  std={s_std:.6f}")

        all_results.append({
            "image": img_path.name,
            "image_size": [img.shape[1], img.shape[0]],
            "precision": precision,
            "n_runs": n_runs,
            "latency_ms": {
                "mean": round(mean_ms, 3),
                "std": round(std_ms, 3),
                "min": round(min_ms, 3),
                "max": round(max_ms, 3),
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3),
                "raw": [round(v, 3) for v in latencies_ms],
            },
            "detection_consistency": {
                "box_counts": box_counts,
                "count_consistent": count_consistent,
                "coord_max_delta_px": round(coord_max_var, 4),
                "coord_consistent": coord_consistent,
            },
        })

    return all_results


def print_comparison_table(fp16_results, fp8_results):
    """Print a side-by-side comparison of FP16 vs FP8 latency stats."""
    print(f"\n{'='*90}")
    print("  LATENCY VARIANCE COMPARISON: FP16 vs FP8")
    print(f"{'='*90}")
    header = (
        f"  {'Image':<35s} "
        f"{'FP16 mean':>10s} {'FP16 std':>9s} {'FP16 p95':>9s} "
        f"{'FP8 mean':>10s} {'FP8 std':>9s} {'FP8 p95':>9s} "
        f"{'Speedup':>8s}"
    )
    print(header)
    print(f"  {'-'*35} {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*9} {'-'*9} {'-'*8}")

    for r16, r8 in zip(fp16_results, fp8_results):
        name = r16["image"][:35]
        m16 = r16["latency_ms"]["mean"]
        s16 = r16["latency_ms"]["std"]
        p16 = r16["latency_ms"]["p95"]
        m8 = r8["latency_ms"]["mean"]
        s8 = r8["latency_ms"]["std"]
        p8 = r8["latency_ms"]["p95"]
        speedup = m16 / m8 if m8 > 0 else 0
        print(
            f"  {name:<35s} "
            f"{m16:>10.2f} {s16:>9.2f} {p16:>9.2f} "
            f"{m8:>10.2f} {s8:>9.2f} {p8:>9.2f} "
            f"{speedup:>7.2f}x"
        )

    # Detection consistency comparison
    print(f"\n  Detection Consistency:")
    print(f"  {'Image':<35s} {'FP16 det':>10s} {'FP8 det':>10s} "
          f"{'FP16 coord':>11s} {'FP8 coord':>11s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*11} {'-'*11}")
    for r16, r8 in zip(fp16_results, fp8_results):
        name = r16["image"][:35]
        c16 = r16["detection_consistency"]
        c8 = r8["detection_consistency"]
        d16 = c16["box_counts"][0] if c16["count_consistent"] else "VARIES"
        d8 = c8["box_counts"][0] if c8["count_consistent"] else "VARIES"
        cv16 = f"{c16['coord_max_delta_px']:.3f}px"
        cv8 = f"{c8['coord_max_delta_px']:.3f}px"
        print(f"  {name:<35s} {str(d16):>10s} {str(d8):>10s} "
              f"{cv16:>11s} {cv8:>11s}")

    print(f"{'='*90}")


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark TensorRT layout detection latency variance"
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
        "--runs", type=int, default=20,
        help="Number of inference runs per image (default: 20)",
    )
    ap.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup runs (default: 5)",
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
    print(f"Runs per image: {args.runs}")
    print(f"Warmup runs: {args.warmup}")

    if args.both:
        fp16_results = run_variance_benchmark(
            "fp16", images, args.runs, args.warmup, args.n_calib
        )
        fp8_results = run_variance_benchmark(
            "fp8", images, args.runs, args.warmup, args.n_calib
        )
        print_comparison_table(fp16_results, fp8_results)

        summary_path = OUTPUT_DIR / "variance_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"fp16": fp16_results, "fp8": fp8_results}, f, indent=2)
        print(f"\nVariance summary saved to: {summary_path}")

    elif args.fp8:
        results = run_variance_benchmark(
            "fp8", images, args.runs, args.warmup, args.n_calib
        )
        summary_path = OUTPUT_DIR / "variance_fp8.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nVariance results saved to: {summary_path}")

    else:
        results = run_variance_benchmark(
            "fp16", images, args.runs, args.warmup, args.n_calib
        )
        summary_path = OUTPUT_DIR / "variance_fp16.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nVariance results saved to: {summary_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
