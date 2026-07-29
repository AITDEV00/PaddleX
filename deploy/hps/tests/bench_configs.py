#!/usr/bin/env python3
"""Benchmark harness for comparing pipeline + batching configurations.

Runs the full async pipeline with a *simulated* GPU model that has
controllable per-batch latency, then measures aggregate throughput,
per-request latency, and GPU call count under concurrent load.

Configurations swept:
  ┌────────────────────────────────────────────────────────────────┐
  │ Config      │ DEPTH │ BATCH │ Description                      │
  ├────────────────────────────────────────────────────────────────┤
  │ serial      │   1   │   1   │ Fully serial (baseline)          │
  │ pipeline    │   3   │   1   │ Pipeline overlap only            │
  │ batch_only  │   1   │   4   │ Batching only                    │
  │ balanced    │   3   │   4   │ Pipeline + batching              │
  │ max_throughput│ 8   │   8   │ Aggressive                       │
  │ wide_batch  │   4   │   8   │ Wide pipeline, big batch         │
  └────────────────────────────────────────────────────────────────┘

The simulated model has a fixed GPU cost per predict() call (regardless
of batch size) plus a small per-image cost, modeling the real-world
situation where batching amortizes kernel-launch overhead:

    latency(batch_n) = FIXED_GPU_MS + PER_IMAGE_MS * n

With FIXED_GPU_MS=40 and PER_IMAGE_MS=5:
    batch=1: 45ms per image → 4 serial images = 180ms
    batch=4: 40 + 20 = 60ms total → 4 images in 60ms  (3× speedup)

Usage:
    cd /home/jyao/ADEO/OCR/PaddleX/deploy/hps
    python3 tests/bench_configs.py
    python3 tests/bench_configs.py --n 100 --fixed 40 --per-image 5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future

import numpy as np

# Ensure the api_compat package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub paddlex so _core.inference imports don't fail (no GPU needed)
if "paddlex" not in sys.modules:
    _stub = type(sys)("paddlex")
    _stub.create_model = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["paddlex"] = _stub

import api_compat._core.inference as inf  # noqa: E402
from api_compat._core.inference import (  # noqa: E402
    _process_batch,
    _process_single,
    state,
)


# ═════════════════════════════════════════════════════════════════════
# Simulated GPU Model
# ═════════════════════════════════════════════════════════════════════


class SimulatedGPUModel:
    """Simulates a GPU model with batch-dependent latency.

    predict(images, batch_size=N) takes:
        FIXED_GPU_MS + PER_IMAGE_MS * N

    This models the real-world behavior where batching amortizes fixed
    kernel-launch and memory-transfer overhead across multiple images.
    """

    def __init__(self, fixed_ms: float, per_image_ms: float):
        self.fixed_ms = fixed_ms
        self.per_image_ms = per_image_ms
        self.call_count = 0
        self.total_images = 0

    def predict(self, images, batch_size=None, **kw):
        if isinstance(images, list):
            n = len(images)
        else:
            n = 1
        self.call_count += 1
        self.total_images += n
        latency = (self.fixed_ms + self.per_image_ms * n) / 1000.0
        time.sleep(latency)
        for _ in range(n):
            yield {"boxes": []}


# ═════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═════════════════════════════════════════════════════════════════════


def _setup_infra(depth: int, batch_size: int, batch_timeout_ms: float,
                 model: SimulatedGPUModel):
    """Configure the inference module with the given parameters."""
    # Override config values
    inf.PIPELINE_DEPTH = depth
    inf.BATCH_SIZE = batch_size
    inf.BATCH_TIMEOUT_MS = batch_timeout_ms

    # Set up state
    state.model = model
    state._task_queue = queue.Queue()
    state._pending = {}
    state._pending_lock = threading.Lock()
    state._load_done = threading.Event()
    state._load_done.set()
    state._load_error = None
    state.semaphore = asyncio.Semaphore(depth)

    return state


def _submit_task(task_id: str) -> Future:
    """Submit a single inference task and return its future."""
    fut: Future = Future()
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    with state._pending_lock:
        state._pending[task_id] = fut
    state._task_queue.put((task_id, img, fut))
    return fut


def _run_worker_once(model: SimulatedGPUModel):
    """Run one iteration of the inference worker loop (process one batch
    or one single task)."""
    if inf.BATCH_SIZE <= 1:
        task = state._task_queue.get()
        if task[0] is None:
            return
        _process_single(task)
    else:
        task = state._task_queue.get()
        if task[0] is None:
            return
        batch = [task]
        deadline = time.monotonic() + inf.BATCH_TIMEOUT_MS / 1000.0
        while len(batch) < inf.BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                extra = state._task_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if extra[0] is None:
                break
            batch.append(extra)
        _process_batch(batch)


async def _benchmark_config(name: str, depth: int, batch_size: int,
                            batch_timeout_ms: float, n: int,
                            fixed_ms: float, per_image_ms: float):
    """Run a single configuration and return metrics dict."""
    model = SimulatedGPUModel(fixed_ms, per_image_ms)
    _setup_infra(depth, batch_size, batch_timeout_ms, model)

    # Start the inference worker in a thread
    worker_stop = threading.Event()

    def worker_loop():
        while not worker_stop.is_set():
            # Check if there's anything to process
            try:
                task = state._task_queue.get(timeout=0.001)
            except queue.Empty:
                continue
            if task[0] is None:
                break
            if inf.BATCH_SIZE <= 1:
                _process_single(task)
            else:
                batch = [task]
                deadline = time.monotonic() + inf.BATCH_TIMEOUT_MS / 1000.0
                while len(batch) < inf.BATCH_SIZE:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        extra = state._task_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if extra[0] is None:
                        break
                    batch.append(extra)
                _process_batch(batch)

    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    # Submit N concurrent requests
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    futures = []
    for i in range(n):
        fut = _submit_task(f"bench_{name}_{i}")
        futures.append(fut)

    # Wait for all results
    latencies = []
    for fut in futures:
        await loop.run_in_executor(None, fut.result)
        latencies.append(time.perf_counter() - t0)

    wall = time.perf_counter() - t0
    worker_stop.set()
    state._task_queue.put((None, None, None))  # signal stop
    t.join(timeout=2)

    latencies.sort()
    rps = n / wall if wall > 0 else 0
    gpu_calls = model.call_count
    avg_latency = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[-1] if len(latencies) < 100 else latencies[int(len(latencies) * 0.99)]

    ideal_serial_ms = (fixed_ms + per_image_ms) * n
    speedup = ideal_serial_ms / (wall * 1000) if wall > 0 else 0

    metrics = {
        "config": name,
        "depth": depth,
        "batch": batch_size,
        "timeout_ms": batch_timeout_ms,
        "n_requests": n,
        "wall_s": wall,
        "rps": rps,
        "gpu_calls": gpu_calls,
        "images_per_call": n / gpu_calls if gpu_calls > 0 else 0,
        "avg_latency_s": avg_latency,
        "p50_s": p50,
        "p95_s": p95,
        "p99_s": p99,
        "ideal_serial_ms": ideal_serial_ms,
        "speedup_vs_serial": speedup,
    }
    return metrics


# ═════════════════════════════════════════════════════════════════════
# Configuration Matrix
# ═════════════════════════════════════════════════════════════════════

CONFIGS = [
    # name,            depth, batch, timeout_ms
    ("serial",           1,     1,     0),
    ("pipeline_d3",      3,     1,     0),
    ("batch_b4",         1,     4,    10),
    ("balanced_d3_b4",   3,     4,    10),
    ("wide_d4_b8",       4,     8,    10),
    ("max_d8_b8",        8,     8,    10),
]


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════


async def run_all(args):
    print(f"\n{'='*80}")
    print(f"  GPU Batching Benchmark — {args.n} concurrent requests")
    print(f"  Simulated GPU: fixed={args.fixed}ms + {args.per_image}ms/image")
    print(f"  Formula: latency(batch_n) = {args.fixed} + {args.per_image} * n  ms")
    print(f"{'='*80}\n")

    all_metrics = []
    for name, depth, batch, timeout in CONFIGS:
        print(f"  Running {name} (depth={depth}, batch={batch}, "
              f"timeout={timeout}ms)...", end=" ", flush=True)
        m = await _benchmark_config(
            name, depth, batch, timeout, args.n,
            args.fixed, args.per_image,
        )
        all_metrics.append(m)
        print(f"done ({m['wall_s']:.3f}s, {m['gpu_calls']} GPU calls)")

    # Print results table
    print(f"\n{'='*100}")
    print("  RESULTS")
    print(f"{'='*100}")
    header = (
        f"{'Config':<18} {'Depth':>5} {'Batch':>5} "
        f"{'Wall(s)':>8} {'RPS':>7} {'GPU':>4} {'Img/Call':>8} "
        f"{'p50(ms)':>8} {'p95(ms)':>8} {'p99(ms)':>8} "
        f"{'Speedup':>8}"
    )
    print(header)
    print("-" * 100)
    for m in all_metrics:
        row = (
            f"{m['config']:<18} {m['depth']:>5} {m['batch']:>5} "
            f"{m['wall_s']:>8.3f} {m['rps']:>7.1f} {m['gpu_calls']:>4} "
            f"{m['images_per_call']:>8.1f} "
            f"{m['p50_s']*1000:>8.1f} {m['p95_s']*1000:>8.1f} "
            f"{m['p99_s']*1000:>8.1f} {m['speedup_vs_serial']:>8.2f}x"
        )
        print(row)
    print("-" * 100)

    # Analysis
    best_rps = max(all_metrics, key=lambda m: m["rps"])
    best_p99 = min(all_metrics, key=lambda m: m["p99_s"])
    best_speedup = max(all_metrics, key=lambda m: m["speedup_vs_serial"])
    serial = next(m for m in all_metrics if m["config"] == "serial")

    print("\n  Analysis:")
    print(f"    Best throughput:    {best_rps['config']:<18} "
          f"({best_rps['rps']:.1f} RPS, "
          f"{best_rps['rps']/serial['rps']:.2f}× vs serial)")
    print(f"    Best p99 latency:   {best_p99['config']:<18} "
          f"({best_p99['p99_s']*1000:.1f}ms)")
    print(f"    Best speedup:       {best_speedup['config']:<18} "
          f"({best_speedup['speedup_vs_serial']:.2f}× vs serial)")

    # GPU call efficiency
    print("\n  GPU Call Efficiency:")
    for m in all_metrics:
        print(f"    {m['config']:<18}  {m['gpu_calls']:>3} calls for "
              f"{m['n_requests']} images "
              f"({m['images_per_call']:.1f} img/call)")

    print(f"\n{'='*100}\n")

    # Recommendation
    print("  Recommendation:")
    balanced = next(
        (m for m in all_metrics if m["config"] == "balanced_d3_b4"), None
    )
    if balanced and balanced["speedup_vs_serial"] > 2.0:
        print(f"    ✓ DEPTH=3 + BATCH=4 provides {balanced['speedup_vs_serial']:.2f}× "
              f"speedup with good latency")
        print(f"      ({balanced['rps']:.1f} RPS, p99={balanced['p99_s']*1000:.1f}ms)")
    if best_rps["config"] != "balanced_d3_b4":
        print(f"    ✓ For max throughput: {best_rps['config']} "
              f"({best_rps['rps']:.1f} RPS)")
    print(f"\n{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU batching configurations"
    )
    parser.add_argument(
        "--n", type=int, default=32,
        help="Number of concurrent requests (default: 32)",
    )
    parser.add_argument(
        "--fixed", type=float, default=40.0,
        help="Fixed GPU cost per predict() call in ms (default: 40)",
    )
    parser.add_argument(
        "--per-image", type=float, default=5.0,
        help="Per-image GPU cost in ms (default: 5)",
    )
    args = parser.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
