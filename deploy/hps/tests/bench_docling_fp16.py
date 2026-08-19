"""Benchmark the docling layer at fp16 using base64 (no image-server bottleneck).

Loads the image once as base64, then fires concurrent /v1/convert/source
requests. Compares against the docling fp8 scratchpad figures and the mistral
fp16 figures to isolate whether the gap is precision or the API layer.
"""
import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def one_req(url: str, b64: str, filename: str) -> float:
    payload = {
        "sources": [
            {
                "kind": "file",
                "filename": filename,
                "base64_string": b64,
            }
        ],
        "options": {"to_formats": ["md"]},
    }
    t0 = time.monotonic()
    r = requests.post(f"{url}/v1/convert/source", json=payload, timeout=60)
    dt = (time.monotonic() - t0) * 1000
    r.raise_for_status()
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081")
    ap.add_argument("--image", default="/home/jyao/ADEO/OCR/PaddleX/deploy/hps/tests/mig_inference/input/business_letter.png")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--serial", type=int, default=10)
    args = ap.parse_args()

    b64 = _b64(args.image)
    fname = args.image.rsplit("/", 1)[-1]
    url = args.url

    # Serial baseline
    lat = [one_req(url, b64, fname) for _ in range(args.serial)]
    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95) - 1]
    total = sum(lat)
    print(f"=== SERIAL: {args.serial} requests ===")
    print(f"  throughput    = {args.serial / (total / 1000):.2f} req/s")
    print(f"  p50 latency   = {p50:.1f} ms")
    print(f"  p95 latency   = {p95:.1f} ms")
    print(f"  avg latency   = {total / len(lat):.1f} ms")

    # Concurrent
    print(f"\n=== CONCURRENT: {args.concurrency} at once x {args.runs} runs ===")
    t_start = time.monotonic()
    total = 0
    for run in range(args.runs):
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(lambda _: one_req(url, b64, fname), range(args.concurrency)))
        total += args.concurrency
    wall = time.monotonic() - t_start
    print(f"  total requests = {total}")
    print(f"  wall-clock     = {wall * 1000:.1f} ms")
    print(f"  throughput     = {total / wall:.2f} req/s")


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


if __name__ == "__main__":
    main()