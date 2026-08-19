#!/usr/bin/env python
"""Benchmark the HPS /v1/ocr endpoint: serial vs concurrent batch.

Uses the official `mistralai` client.

Two comparisons:
  1. SERIAL: send 1 image repeatedly, back-to-back (measures single-request
     latency & serial throughput).
  2. CONCURRENT: send N images at once via a thread pool (true concurrency).
     This is what triggers the direct backend's micro-batching
     (batch_size=4, batch_timeout_ms=3) so GPU requests get coalesced.

Usage:
    python tests/bench_mistral_ocr.py --url http://localhost:8080
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from mistralai.client import Mistral
from mistralai.client.models import DocumentURLChunk

IMAGES = [
    "http://localhost:9090/business_letter.png",
    "http://localhost:9090/table.png",
    "http://localhost:9090/receipt.png",
    "http://localhost:9090/form_fields.png",
    "http://localhost:9090/dense_text.png",
    "http://localhost:9090/multi_language.png",
]


def one_request(client: Mistral, url: str) -> float:
    t0 = time.perf_counter()
    client.ocr.process(model="PP-DocLayoutV3",
                       document=DocumentURLChunk(document_url=url))
    return time.perf_counter() - t0


def bench_serial(client: Mistral, runs: int) -> list[float]:
    return [one_request(client, IMAGES[0]) for _ in range(runs)]


def bench_concurrent(client: Mistral, n: int, runs: int) -> list[list[float]]:
    """Each run: dispatch n requests concurrently (1 per thread). Returns per-run latency lists."""
    results: list[list[float]] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        for _ in range(runs):
            urls = [IMAGES[i % len(IMAGES)] for i in range(n)]
            lat = list(ex.map(lambda u: one_request(client, u), urls))
            results.append(lat)
    return results


def stats(name: str, per_req: list[float], n: int, wall: float) -> None:
    per_req.sort()
    p50 = per_req[len(per_req) // 2]
    p95 = per_req[int(len(per_req) * 0.95) - 1]
    print(f"{name}:")
    print(f"  requests      = {n}")
    print(f"  wall-clock    = {wall * 1000:8.1f} ms")
    print(f"  throughput    = {n / wall:8.2f} req/s")
    print(f"  p50 latency   = {p50 * 1000:8.1f} ms")
    print(f"  p95 latency   = {p95 * 1000:8.1f} ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--images", default=8, type=int, help="concurrent requests per batch")
    ap.add_argument("--serial-runs", default=10, type=int)
    ap.add_argument("--concurrent-runs", default=5, type=int)
    args = ap.parse_args()

    client = Mistral(api_key="local-test", server_url=args.url)
    n = args.images

    # Warmup (load engine, JIT, etc.)
    one_request(client, IMAGES[0])

    # SERIAL
    t0 = time.perf_counter()
    serial_lat = bench_serial(client, args.serial_runs)
    serial_wall = time.perf_counter() - t0
    print(f"=== SERIAL: {args.serial_runs} back-to-back requests ===")
    stats("serial", serial_lat, args.serial_runs, serial_wall)
    print(f"  avg single latency = {statistics.mean(serial_lat) * 1000:.1f} ms")
    print()

    # CONCURRENT
    print(f"=== CONCURRENT: {n} requests at once x {args.concurrent_runs} runs ===")
    runs = bench_concurrent(client, n, args.concurrent_runs)
    all_per_req = [lat for run in runs for lat in run]
    walls = [max(run) for run in runs]  # wall = slowest request in each run
    total_wall = sum(walls)
    stats("concurrent", all_per_req, n * args.concurrent_runs, total_wall)
    print(f"  avg run wall-clock = {statistics.mean(walls) * 1000:.1f} ms")
    print(f"  per-request p50    = {statistics.median(sorted(all_per_req)) * 1000:.1f} ms")
    print()

    # SPEEDUP
    serial_throughput = args.serial_runs / serial_wall
    conc_throughput = (n * args.concurrent_runs) / total_wall
    print(f"=== THROUGHPUT ===")
    print(f"  serial     = {serial_throughput:6.2f} req/s")
    print(f"  concurrent = {conc_throughput:6.2f} req/s")
    print(f"  speedup    = {conc_throughput / serial_throughput:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())