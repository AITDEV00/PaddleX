#!/usr/bin/env python
"""FAIR side-by-side benchmark: mistral /v1/ocr vs docling /v1/convert/source.

Both use the SAME image URL fetched over HTTP by each server (identical
transport) so the only difference is the API layer's own cost (serialize,
convert, response size). Requires a threaded image server serving the test
image (e.g. tests/threaded_file_server.py on :9090).

Usage:
    python tests/bench_fair_compare.py \
        --mistral http://localhost:8080 \
        --docling http://localhost:8081 \
        --image-url http://localhost:9090/business_letter.png
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def mistral_req(url: str, image_url: str) -> tuple[float, int]:
    payload = {
        "model": "PP-DocLayoutV3",
        "document": {"type": "document_url", "document_url": image_url},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/ocr", json=payload, timeout=60)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt, len(r.content)


def docling_req(url: str, image_url: str) -> tuple[float, int]:
    payload = {
        "sources": [{"kind": "http", "url": image_url}],
        "options": {"to_formats": ["md"]},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/convert/source", json=payload, timeout=60)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt, len(r.content)


def run_concurrent(fn, url, image_url, n: int, runs: int) -> dict:
    walls: list[float] = []
    all_lat: list[float] = []
    total_bytes = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(lambda _: fn(url, image_url), range(n)))
        walls.append(time.perf_counter() - t0)
        all_lat.extend(r[0] for r in results)
        total_bytes += sum(r[1] for r in results)
    total_req = n * runs
    all_lat.sort()
    p95 = all_lat[int(len(all_lat) * 0.95) - 1] if len(all_lat) else 0.0
    return {
        "throughput": total_req / sum(walls),
        "mean_ms": statistics.mean(all_lat) * 1000,
        "median_ms": statistics.median(all_lat) * 1000,
        "p95_ms": p95 * 1000,
        "resp_bytes": total_bytes / total_req,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mistral", default="http://localhost:8080")
    ap.add_argument("--docling", default="http://localhost:8081")
    ap.add_argument("--image-url", default="http://localhost:9090/business_letter.png")
    ap.add_argument("--concurrency", default="1,5,10,20,40", help="comma-separated")
    ap.add_argument("--runs", default=5, type=int)
    args = ap.parse_args()

    image_url = args.image_url
    print(f"image url = {image_url}\n")

    conc_levels = [int(x) for x in args.concurrency.split(",")]

    for n in conc_levels:
        # warmup each endpoint
        try:
            mistral_req(args.mistral, image_url)
        except Exception:
            pass
        try:
            docling_req(args.docling, image_url)
        except Exception:
            pass

        # Interleave A/B per round to cancel GPU clock drift / ordering bias.
        m_agg: dict[str, list[float]] = {"throughput": [], "mean_ms": [], "median_ms": [], "p95_ms": [], "resp_bytes": []}
        d_agg: dict[str, list[float]] = {"throughput": [], "mean_ms": [], "median_ms": [], "p95_ms": [], "resp_bytes": []}
        for _ in range(args.runs):
            m = run_concurrent(mistral_req, args.mistral, image_url, n, 1)
            d = run_concurrent(docling_req, args.docling, image_url, n, 1)
            for k in m_agg:
                m_agg[k].append(m[k])
                d_agg[k].append(d[k])
        mm = {k: statistics.mean(v) for k, v in m_agg.items()}
        dd = {k: statistics.mean(v) for k, v in d_agg.items()}

        print(f"--- Concurrency {n} (runs={args.runs}) ---")
        print(f"  {'':12} {'mistral':>12} {'docling':>12} {'Δ':>8}")
        print(f"  {'throughput':<12} {mm['throughput']:12.1f} {dd['throughput']:12.1f} "
              f"{mm['throughput'] - dd['throughput']:+.1f} req/s")
        print(f"  {'mean lat':<12} {mm['mean_ms']:12.1f} {dd['mean_ms']:12.1f} "
              f"{mm['mean_ms'] - dd['mean_ms']:+.1f} ms")
        print(f"  {'median lat':<12} {mm['median_ms']:12.1f} {dd['median_ms']:12.1f} "
              f"{mm['median_ms'] - dd['median_ms']:+.1f} ms")
        print(f"  {'p95 lat':<12} {mm['p95_ms']:12.1f} {dd['p95_ms']:12.1f} ms")
        print(f"  {'resp bytes':<12} {mm['resp_bytes']:12.0f} {dd['resp_bytes']:12.0f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())