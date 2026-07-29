#!/usr/bin/env python3
"""Concurrency stress test for the HPS Docling API.

Sends concurrent layout-detection requests at various concurrency levels
and reports latency percentiles (p50/p95/p99), throughput, and server-side
processing times.

Usage (run INSIDE the container):
    python3 /tmp/stress_test.py
    python3 /tmp/stress_test.py --concurrency 1,5,10,20,50 --rounds 3
    python3 /tmp/stress_test.py --url http://localhost:8080 --image /tmp/book.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp

# ─── Helpers ────────────────────────────────────────────────────────────────


def percentile(data: list[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_ms(ms: float) -> str:
    return f"{ms:8.1f}ms"


def fmt_rps(rps: float) -> str:
    return f"{rps:7.1f} r/s"


# ─── Single request ─────────────────────────────────────────────────────────


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    image_data: bytes,
    filename: str,
    req_id: int,
) -> dict:
    """Send one /v1/convert/file request, return timing info."""
    form = aiohttp.FormData()
    form.add_field("file", image_data, filename=filename, content_type="image/jpeg")

    t0 = time.perf_counter()
    try:
        async with session.post(url, data=form) as resp:
            body = await resp.read()
            t1 = time.perf_counter()

            wall_ms = (t1 - t0) * 1000.0
            status = resp.status

            server_time = None
            layout_time = None
            if status == 200:
                try:
                    d = json.loads(body)
                    server_time = d.get("processing_time")
                    timings = d.get("timings", {})
                    layout = timings.get("layout", {})
                    times = layout.get("times", [])
                    if times:
                        layout_time = sum(times) * 1000  # seconds → ms
                except Exception:
                    pass

            return {
                "req_id": req_id,
                "status": status,
                "wall_ms": wall_ms,
                "server_time_s": server_time,
                "layout_time_ms": layout_time,
                "error": None if status == 200 else body[:200].decode("utf-8", errors="replace"),
            }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "req_id": req_id,
            "status": 0,
            "wall_ms": (t1 - t0) * 1000.0,
            "server_time_s": None,
            "layout_time_ms": None,
            "error": str(e),
        }


# ─── Concurrency test ───────────────────────────────────────────────────────


async def run_concurrency_level(
    url: str,
    image_data: bytes,
    filename: str,
    concurrency: int,
    num_requests: int,
    round_num: int,
) -> dict:
    """Send num_requests at the given concurrency level."""

    connector = aiohttp.TCPConnector(limit=concurrency + 10, force_close=False)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict] = []
        completed = 0

        async def worker(req_id: int):
            nonlocal completed
            async with semaphore:
                r = await send_request(session, url, image_data, filename, req_id)
                results.append(r)
                completed += 1
                if completed % max(1, num_requests // 10) == 0:
                    print(f"    [{completed}/{num_requests}] done", flush=True)

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    wall_total = t_end - t_start

    # Sort results by req_id for consistency
    results.sort(key=lambda r: r["req_id"])

    # Compute stats
    wall_times = [r["wall_ms"] for r in results if r["status"] == 200]
    server_times = [r["server_time_s"] * 1000 for r in results if r["server_time_s"] is not None]
    layout_times = [r["layout_time_ms"] for r in results if r["layout_time_ms"] is not None]
    errors = [r for r in results if r["status"] != 200]

    success = len(wall_times)
    throughput = success / wall_total if wall_total > 0 else 0

    stats = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "round": round_num,
        "success": success,
        "errors": len(errors),
        "wall_total_s": wall_total,
        "throughput_rps": throughput,
        "wall_ms": {
            "min": min(wall_times) if wall_times else 0,
            "p50": percentile(wall_times, 50) if wall_times else 0,
            "p95": percentile(wall_times, 95) if wall_times else 0,
            "p99": percentile(wall_times, 99) if wall_times else 0,
            "max": max(wall_times) if wall_times else 0,
            "mean": statistics.mean(wall_times) if wall_times else 0,
        },
        "server_ms": {
            "min": min(server_times) if server_times else 0,
            "p50": percentile(server_times, 50) if server_times else 0,
            "p95": percentile(server_times, 95) if server_times else 0,
            "p99": percentile(server_times, 99) if server_times else 0,
            "max": max(server_times) if server_times else 0,
            "mean": statistics.mean(server_times) if server_times else 0,
        },
        "layout_ms": {
            "min": min(layout_times) if layout_times else 0,
            "p50": percentile(layout_times, 50) if layout_times else 0,
            "p95": percentile(layout_times, 95) if layout_times else 0,
            "p99": percentile(layout_times, 99) if layout_times else 0,
            "max": max(layout_times) if layout_times else 0,
            "mean": statistics.mean(layout_times) if layout_times else 0,
        },
    }

    return stats


def print_stats(stats: dict):
    c = stats["concurrency"]
    n = stats["num_requests"]
    r = stats["round"]
    w = stats["wall_ms"]
    s = stats["server_ms"]
    l = stats["layout_ms"]
    tput = stats["throughput_rps"]
    errs = stats["errors"]

    print(f"\n  Concurrency={c}  Requests={n}  Round={r}  Errors={errs}")
    print(f"  {'Metric':<20} {'min':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'max':>10} {'mean':>10}")
    print(f"  {'─'*80}")
    print(f"  {'Wall (end-to-end)':<20} {fmt_ms(w['min']):>10} {fmt_ms(w['p50']):>10} {fmt_ms(w['p95']):>10} {fmt_ms(w['p99']):>10} {fmt_ms(w['max']):>10} {fmt_ms(w['mean']):>10}")
    if s["p50"] > 0:
        print(f"  {'Server processing':<20} {fmt_ms(s['min']):>10} {fmt_ms(s['p50']):>10} {fmt_ms(s['p95']):>10} {fmt_ms(s['p99']):>10} {fmt_ms(s['max']):>10} {fmt_ms(s['mean']):>10}")
    if l["p50"] > 0:
        print(f"  {'Layout (GPU)':<20} {fmt_ms(l['min']):>10} {fmt_ms(l['p50']):>10} {fmt_ms(l['p95']):>10} {fmt_ms(l['p99']):>10} {fmt_ms(l['max']):>10} {fmt_ms(l['mean']):>10}")
    print(f"  {'Throughput':<20} {fmt_rps(tput):>10}")
    print(f"  {'Total wall time':<20} {stats['wall_total_s']:>10.2f}s")


def print_summary(all_stats: list[dict], image_name: str):
    """Print a summary table aggregating across rounds."""
    # Group by concurrency
    by_conc: dict[int, list[dict]] = {}
    for s in all_stats:
        by_conc.setdefault(s["concurrency"], []).append(s)

    print("\n" + "=" * 90)
    print(f"  SUMMARY — Image: {image_name}")
    print("=" * 90)
    print(f"  {'Conc':>5}  {'N':>5}  {'Rounds':>6}  {'Errors':>6}  {'Wall p50':>10}  {'Wall p95':>10}  {'Wall p99':>10}  {'Server p50':>11}  {'Server p99':>11}  {'Throughput':>10}")
    print(f"  {'─'*85}")

    for conc in sorted(by_conc.keys()):
        rounds = by_conc[conc]
        n = rounds[0]["num_requests"]
        num_rounds = len(rounds)
        total_errs = sum(r["errors"] for r in rounds)

        # Average across rounds
        avg_wall_p50 = statistics.mean(r["wall_ms"]["p50"] for r in rounds)
        avg_wall_p95 = statistics.mean(r["wall_ms"]["p95"] for r in rounds)
        avg_wall_p99 = statistics.mean(r["wall_ms"]["p99"] for r in rounds)
        avg_server_p50 = statistics.mean(r["server_ms"]["p50"] for r in rounds)
        avg_server_p99 = statistics.mean(r["server_ms"]["p99"] for r in rounds)
        avg_tput = statistics.mean(r["throughput_rps"] for r in rounds)

        print(f"  {conc:>5}  {n:>5}  {num_rounds:>6}  {total_errs:>6}  {fmt_ms(avg_wall_p50):>10}  {fmt_ms(avg_wall_p95):>10}  {fmt_ms(avg_wall_p99):>10}  {fmt_ms(avg_server_p50):>11}  {fmt_ms(avg_server_p99):>11}  {fmt_rps(avg_tput):>10}")

    print()


# ─── Main ───────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="HPS API concurrency stress test")
    parser.add_argument("--url", default="http://localhost:8080/v1/convert/file", help="API endpoint URL")
    parser.add_argument("--image", default="/tmp/book.jpg", help="Image file to send")
    parser.add_argument("--concurrency", default="1,2,5,10,20,50", help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=0, help="Requests per round (0 = 10× concurrency, min 10)")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per concurrency level")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests before timing")
    args = parser.parse_args()

    # Load image
    with open(args.image, "rb") as f:
        image_data = f.read()
    image_name = args.image.split("/")[-1]
    print(f"Image: {args.image} ({len(image_data)} bytes)")

    # Parse concurrency levels
    conc_levels = [int(x) for x in args.concurrency.split(",")]
    print(f"Concurrency levels: {conc_levels}")
    print(f"Rounds per level: {args.rounds}")
    print(f"Endpoint: {args.url}")

    # Warmup
    if args.warmup > 0:
        print(f"\nWarmup: {args.warmup} sequential requests...")
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(args.warmup):
                r = await send_request(session, args.url, image_data, image_name, i)
                print(f"  warmup {i+1}/{args.warmup}: status={r['status']} wall={r['wall_ms']:.1f}ms")
        print("Warmup complete.\n")

    all_stats = []

    for conc in conc_levels:
        for rnd in range(1, args.rounds + 1):
            n = args.requests if args.requests > 0 else max(10, conc * 10)
            print(f"\n▶ Concurrency={conc}  Round {rnd}/{args.rounds}  ({n} requests)")

            stats = await run_concurrency_level(args.url, image_data, image_name, conc, n, rnd)
            print_stats(stats)
            all_stats.append(stats)

            # Brief cooldown between rounds
            await asyncio.sleep(1.0)

    # Summary
    print_summary(all_stats, image_name)

    # Save raw results
    with open("/tmp/stress_results.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("Raw results saved to /tmp/stress_results.json")


if __name__ == "__main__":
    asyncio.run(main())
