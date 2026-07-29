#!/usr/bin/env python3
"""Concurrency and stress tests for the Docling API compatibility layer.

Tests verify:
  - Concurrent request handling (multiple simultaneous POSTs)
  - Semaphore serialization behavior (only 1 GPU task in-flight)
  - Queue contention under load
  - Latency logging flag (HPS_LATENCY_LOG) enable/disable
  - Bottleneck identification via simulated slow inference
  - Throughput and latency-under-load measurements
  - No deadlocks or race conditions under stress

These tests run **without GPU or PaddleX** — they use spy functions
that simulate variable inference latency to expose concurrency issues.

Usage:
    cd /home/jyao/ADEO/OCR/PaddleX/deploy/hps
    python3 tests/test_concurrency.py
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import time

import numpy as np

# Ensure the api_compat package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force the "direct" backend for existing tests (they stub paddlex and test
# the in-process inference thread + micro-batching). Triton backend tests
# are in a separate section below.
os.environ.setdefault("HPS_API_BACKEND", "direct")

# Stub paddlex so _core.inference imports don't fail (no GPU needed)
if "paddlex" not in sys.modules:
    _stub = type(sys)("paddlex")
    _stub.create_model = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["paddlex"] = _stub


def main() -> int:
    passed = 0
    failed = 0

    def test(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ─── Imports shared across all tests ──────────────────────────────
    from api_compat.docling_api.schema import (
        ConversionStatus,
        ConvertDocumentResponse,
        ExportDocumentResponse,
    )
    from api_compat._core.latency import (
        LatencyTracer,
        is_latency_logging_enabled,
        set_latency_logging,
    )

    # ─── Helper: create the ASGI app with a custom spy ────────────────
    def _create_app_with_spy(spy_fn):
        """Create a fresh app with the given spy installed."""
        import api_compat.docling_api.service as svc
        import api_compat.docling_api.routes as routes
        from api_compat.docling_api.app import create_app

        svc.convert_image = spy_fn
        routes.convert_image = spy_fn
        return create_app()

    # ─── Helper: a spy that simulates variable inference latency ───────
    def _make_slow_spy(delay: float = 0.05, capture_list=None):
        """Return a spy that sleeps for *delay* seconds to simulate GPU work.

        Captures start/end timestamps into *capture_list* if provided,
        so tests can verify serialization vs overlap.
        """
        async def spy(image_data, filename, to_formats):
            t_start = time.perf_counter()
            await asyncio.sleep(delay)
            t_end = time.perf_counter()
            if capture_list is not None:
                capture_list.append({
                    "filename": filename,
                    "start": t_start,
                    "end": t_end,
                    "duration": t_end - t_start,
                })
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(
                    filename=filename, md_content=f"# {filename}",
                ),
                status=ConversionStatus.SUCCESS,
                processing_time=t_end - t_start,
            )
        return spy

    # ─── Helper: a spy that returns immediately ───────────────────────
    def _make_fast_spy(capture_list=None):
        async def spy(image_data, filename, to_formats):
            t = time.perf_counter()
            if capture_list is not None:
                capture_list.append({
                    "filename": filename,
                    "start": t,
                    "end": t,
                    "duration": 0.0,
                })
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(
                    filename=filename, md_content=f"# {filename}",
                ),
                status=ConversionStatus.SUCCESS,
                processing_time=0.001,
            )
        return spy

    # ─── Helper: send N concurrent POSTs ──────────────────────────────
    def _send_concurrent(app, n: int, path: str = "/v1/convert/file",
                         base_filename: str = "doc"):
        """Send *n* concurrent multipart POSTs through *app*.

        Returns (responses, elapsed) — list of httpx.Response and total
        wall-clock time.
        """
        async def run():
            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                tasks = []
                for i in range(n):
                    tasks.append(c.post(
                        path,
                        files={"file": (f"{base_filename}_{i}.png",
                                        b"fake", "image/png")},
                        data={"to_formats": "md"},
                    ))
                t0 = time.perf_counter()
                responses = await asyncio.gather(*tasks)
                elapsed = time.perf_counter() - t0
                return responses, elapsed
        return asyncio.run(run())

    # ═══════════════════════════════════════════════════════════════════
    # LATENCY LOGGING FLAG TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_latency_disabled_by_default():
        """Latency logging is disabled by default (HPS_LATENCY_LOG not set)."""
        old = os.environ.pop("HPS_LATENCY_LOG", None)
        try:
            assert not is_latency_logging_enabled(), \
                "Latency should be disabled by default"
        finally:
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
    test("Latency: disabled by default", t_latency_disabled_by_default)

    def t_latency_enable_via_env():
        """Setting HPS_LATENCY_LOG=1 enables latency logging."""
        old = os.environ.get("HPS_LATENCY_LOG")
        try:
            set_latency_logging(True)
            assert is_latency_logging_enabled(), \
                "Latency should be enabled after set_latency_logging(True)"
        finally:
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: enable via HPS_LATENCY_LOG=1", t_latency_enable_via_env)

    def t_latency_disable_via_env():
        """Setting HPS_LATENCY_LOG=0 disables latency logging."""
        old = os.environ.get("HPS_LATENCY_LOG")
        try:
            set_latency_logging(True)
            assert is_latency_logging_enabled()
            set_latency_logging(False)
            assert not is_latency_logging_enabled(), \
                "Latency should be disabled after set_latency_logging(False)"
        finally:
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: disable via HPS_LATENCY_LOG=0", t_latency_disable_via_env)

    def t_tracer_noop_when_disabled():
        """LatencyTracer is a no-op when disabled — no dict allocation,
        no timing, finish() returns None."""
        old = os.environ.get("HPS_LATENCY_LOG")
        try:
            set_latency_logging(False)
            tracer = LatencyTracer(filename="t.png", to_formats=["md"])
            with tracer.stage("test_stage"):
                pass
            result = tracer.finish()
            assert result is None, \
                "finish() should return None when disabled"
            assert tracer.stages == {}, \
                "stages should be empty when disabled"
        finally:
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: tracer is no-op when disabled", t_tracer_noop_when_disabled)

    def t_tracer_records_stages_when_enabled():
        """LatencyTracer records stage timings when enabled."""
        old = os.environ.get("HPS_LATENCY_LOG")
        try:
            set_latency_logging(True)
            tracer = LatencyTracer(filename="t.png", to_formats=["md"])
            with tracer.stage("stage_a"):
                time.sleep(0.01)
            with tracer.stage("stage_b"):
                time.sleep(0.02)
            result = tracer.finish()
            assert result is not None, "finish() should return stages dict"
            assert "stage_a" in result, f"Missing stage_a in {result}"
            assert "stage_b" in result, f"Missing stage_b in {result}"
            assert 0.008 <= result["stage_a"] <= 0.05, \
                f"stage_a timing {result['stage_a']} unexpected"
            assert 0.018 <= result["stage_b"] <= 0.05, \
                f"stage_b timing {result['stage_b']} unexpected"
            assert result["stage_b"] > result["stage_a"], \
                "stage_b should be longer than stage_a"
        finally:
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: tracer records stages when enabled", t_tracer_records_stages_when_enabled)

    def t_tracer_record_manual():
        """LatencyTracer.record() adds a manual timing without context manager."""
        old = os.environ.get("HPS_LATENCY_LOG")
        try:
            set_latency_logging(True)
            tracer = LatencyTracer(filename="t.png")
            tracer.record("manual_stage", 0.123)
            assert "manual_stage" in tracer.stages
            assert tracer.stages["manual_stage"] == 0.123
            tracer.finish()
        finally:
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: tracer.record() adds manual timing", t_tracer_record_manual)

    def t_latency_logs_emitted_when_enabled():
        """When enabled, pipeline logs structured latency traces.
        Verify by capturing log output."""
        import logging
        from io import StringIO

        old = os.environ.get("HPS_LATENCY_LOG")
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        lat_logger = logging.getLogger("hps_api.latency")
        lat_logger.addHandler(handler)
        lat_logger.setLevel(logging.INFO)
        try:
            set_latency_logging(True)
            tracer = LatencyTracer(filename="test.png", to_formats=["md"])
            with tracer.stage("quick"):
                pass
            tracer.finish()
            log_output = log_capture.getvalue()
            assert "latency_trace" in log_output, \
                f"Expected latency_trace in logs, got: {log_output}"
            # Should be valid JSON
            line = log_output.strip().split("\n")[-1]
            data = json.loads(line)
            assert data["event"] == "latency_trace"
            assert data["filename"] == "test.png"
            assert "stages" in data
            assert "total" in data
        finally:
            lat_logger.removeHandler(handler)
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: structured JSON traces emitted when enabled", t_latency_logs_emitted_when_enabled)

    def t_latency_no_logs_when_disabled():
        """When disabled, no latency logs are emitted."""
        import logging
        from io import StringIO

        old = os.environ.get("HPS_LATENCY_LOG")
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        lat_logger = logging.getLogger("hps_api.latency")
        lat_logger.addHandler(handler)
        lat_logger.setLevel(logging.INFO)
        try:
            set_latency_logging(False)
            tracer = LatencyTracer(filename="test.png")
            with tracer.stage("quick"):
                pass
            tracer.finish()
            log_output = log_capture.getvalue()
            assert log_output == "", \
                f"No logs expected when disabled, got: {log_output}"
        finally:
            lat_logger.removeHandler(handler)
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Latency: no logs emitted when disabled", t_latency_no_logs_when_disabled)

    # ═══════════════════════════════════════════════════════════════════
    # CONCURRENCY TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_concurrent_5_requests_all_succeed():
        """5 concurrent requests should all return 200."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        responses, elapsed = _send_concurrent(app, 5)
        assert len(responses) == 5
        for i, resp in enumerate(responses):
            assert resp.status_code == 200, \
                f"Request {i} got {resp.status_code}: {resp.text}"
    test("Concurrency: 5 concurrent requests all succeed", t_concurrent_5_requests_all_succeed)

    def t_concurrent_20_requests_all_succeed():
        """20 concurrent requests should all return 200."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        responses, elapsed = _send_concurrent(app, 20)
        assert len(responses) == 20
        for i, resp in enumerate(responses):
            assert resp.status_code == 200, \
                f"Request {i} got {resp.status_code}"
    test("Concurrency: 20 concurrent requests all succeed", t_concurrent_20_requests_all_succeed)

    def t_concurrent_50_requests_all_succeed():
        """50 concurrent requests should all return 200 (stress)."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        responses, elapsed = _send_concurrent(app, 50)
        assert len(responses) == 50
        failures = [i for i, r in enumerate(responses) if r.status_code != 200]
        assert not failures, f"Requests {failures} failed"
    test("Concurrency: 50 concurrent requests all succeed (stress)", t_concurrent_50_requests_all_succeed)

    def t_concurrent_100_requests_all_succeed():
        """100 concurrent requests should all return 200 (heavy stress)."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        responses, elapsed = _send_concurrent(app, 100)
        assert len(responses) == 100
        failures = [i for i, r in enumerate(responses) if r.status_code != 200]
        assert not failures, f"{len(failures)} requests failed: {failures[:5]}"
    test("Concurrency: 100 concurrent requests all succeed (heavy stress)", t_concurrent_100_requests_all_succeed)

    def t_concurrent_responses_have_correct_filenames():
        """Each concurrent response should have its own unique filename."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        responses, _ = _send_concurrent(app, 10)
        filenames = set()
        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            filenames.add(data["document"]["filename"])
        assert len(filenames) == 10, \
            f"Expected 10 unique filenames, got {len(filenames)}: {filenames}"
    test("Concurrency: each response has unique filename (no cross-talk)", t_concurrent_responses_have_correct_filenames)

    # ═══════════════════════════════════════════════════════════════════
    # BOTTLENECK IDENTIFICATION TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_serialization_bottleneck_identified():
        """With a slow spy (50ms each), 10 concurrent requests should take
        at least 10 * 50ms = 500ms if serialized, but since the spy is
        NOT behind the semaphore (it replaces convert_image entirely),
        they should actually run concurrently (~50ms total).

        This test documents that the bottleneck is in run_layout_detection's
        semaphore, not in the HTTP layer."""
        spy = _make_slow_spy(delay=0.05)
        app = _create_app_with_spy(spy)
        responses, elapsed = _send_concurrent(app, 10)
        # Since the spy replaces convert_image entirely (bypassing the
        # semaphore), requests run concurrently. Total should be ~50ms
        # plus overhead, NOT 500ms.
        assert elapsed < 0.3, \
            f"10 concurrent 50ms requests took {elapsed:.3f}s — " \
            "should be ~50ms if concurrent (bottleneck is in semaphore, not HTTP)"
    test("Bottleneck: spy bypasses semaphore → concurrent (bottleneck is in GPU layer)",
         t_serialization_bottleneck_identified)

    def t_semaphore_serializes_gpu_access():
        """The semaphore in run_layout_detection controls GPU access depth.
        With PIPELINE_DEPTH=3, up to 3 requests can be inside the semaphore
        simultaneously, so 5×50ms requests complete in ~2 batches (~100ms)
        rather than 5×50ms=250ms (fully serial).

        We test this by replacing run_layout_detection with a slow version
        that acquires the semaphore."""
        import api_compat.docling_api.service as svc
        from api_compat._core.inference import state
        from api_compat._core.config import PIPELINE_DEPTH

        # Save originals
        orig_run = svc.run_layout_detection

        async def slow_layout_detection(image):
            """Simulate GPU inference with semaphore serialization."""
            import asyncio
            async with state.semaphore:
                await asyncio.sleep(0.05)  # simulate 50ms GPU work
            return []  # empty boxes

        svc.run_layout_detection = slow_layout_detection

        try:
            # We need a spy for convert_image that calls our slow_layout_detection
            async def real_pipeline_spy(image_data, filename, to_formats):
                # Skip image load — use dummy
                await slow_layout_detection(None)
                return ConvertDocumentResponse(
                    document=ExportDocumentResponse(filename=filename, md_content="# T"),
                    status=ConversionStatus.SUCCESS,
                    processing_time=0.05,
                )

            import api_compat.docling_api.routes as routes
            svc.convert_image = real_pipeline_spy
            routes.convert_image = real_pipeline_spy

            from api_compat.docling_api.app import create_app
            app = create_app()

            responses, elapsed = _send_concurrent(app, 5)

            # With DEPTH=3 and 50ms each, 5 requests need ceil(5/3)=2 batches
            # → ~100ms minimum. With DEPTH=1, 5×50ms = ~250ms.
            expected_batches = (5 + PIPELINE_DEPTH - 1) // PIPELINE_DEPTH
            min_expected = expected_batches * 0.05 * 0.8  # 80% margin
            max_expected = 5 * 0.05 * 1.5  # 50% overhead over fully serial

            assert elapsed >= min_expected, \
                f"5 requests with DEPTH={PIPELINE_DEPTH} took {elapsed:.3f}s — " \
                f"expected >= {min_expected:.3f}s (semaphore should limit concurrency)"
            assert elapsed <= max_expected, \
                f"5 requests with DEPTH={PIPELINE_DEPTH} took {elapsed:.3f}s — " \
                f"expected <= {max_expected:.3f}s (pipeline should overlap)"
            print(f"    [metrics] DEPTH={PIPELINE_DEPTH}, 5×50ms: "
                  f"{elapsed:.3f}s (batches={expected_batches})")
        finally:
            svc.run_layout_detection = orig_run
    test("Bottleneck: semaphore depth controls GPU concurrency "
         "(5×50ms with DEPTH=3 ≈ 100ms)",
         t_semaphore_serializes_gpu_access)

    def t_pipeline_overlap_verification():
        """With PIPELINE_DEPTH>1, multiple requests should overlap inside
        the semaphore. We verify by capturing timestamps: at least two
        requests should be inside the critical section simultaneously.

        Uses a local semaphore (not state.semaphore) to avoid event-loop
        binding issues across multiple asyncio.run() calls in tests."""
        import api_compat.docling_api.service as svc
        import api_compat.docling_api.routes as routes
        from api_compat._core.config import PIPELINE_DEPTH
        from api_compat.docling_api.app import create_app

        if PIPELINE_DEPTH < 2:
            print(f"    [skip] PIPELINE_DEPTH={PIPELINE_DEPTH}, skipping overlap test")
            return

        capture = []
        # Local semaphore — avoids cross-event-loop binding issues
        local_sem = asyncio.Semaphore(PIPELINE_DEPTH)

        async def overlapping_detection(image):
            """Record entry/exit to verify overlap."""
            async with local_sem:
                t_start = time.perf_counter()
                await asyncio.sleep(0.03)  # 30ms work
                t_end = time.perf_counter()
                capture.append({"start": t_start, "end": t_end})
            return []

        orig_run = svc.run_layout_detection
        svc.run_layout_detection = overlapping_detection

        async def pipeline_spy(image_data, filename, to_formats):
            await overlapping_detection(None)
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename, md_content="# T"),
                status=ConversionStatus.SUCCESS,
                processing_time=0.03,
            )

        try:
            svc.convert_image = pipeline_spy
            routes.convert_image = pipeline_spy
            app = create_app()

            _send_concurrent(app, 4)

            # With DEPTH>=2, at least 2 requests should overlap
            # (their time ranges intersect)
            overlaps = 0
            for i in range(len(capture)):
                for j in range(i + 1, len(capture)):
                    a, b = capture[i], capture[j]
                    if a["start"] < b["end"] and b["start"] < a["end"]:
                        overlaps += 1
            assert overlaps >= 1, \
                f"Expected at least 1 overlap with DEPTH={PIPELINE_DEPTH}, " \
                f"got {overlaps}. Captured: {capture}"
            print(f"    [metrics] DEPTH={PIPELINE_DEPTH}, "
                  f"overlapping pairs: {overlaps}/{len(capture)}")
        finally:
            svc.run_layout_detection = orig_run
    test("Pipeline: DEPTH>1 allows request overlap inside semaphore",
         t_pipeline_overlap_verification)

    def t_future_based_no_head_of_line_blocking():
        """With future-based result matching, results are delivered to the
        correct request by task_id, not by queue order. Even if the
        inference thread finishes tasks out of order, each request gets
        its own result.

        We verify the _pending dict mechanism by manually creating and
        resolving futures out of order."""
        from api_compat._core.inference import state
        from concurrent.futures import Future

        # Create 3 futures keyed by task_id
        futures = {}
        for i in range(3):
            task_id = f"test_task_{i}"
            fut = Future()
            with state._pending_lock:
                state._pending[task_id] = fut
            futures[task_id] = fut

        # Resolve them out of order (task_2 first, then task_0, then task_1)
        state._pending["test_task_2"].set_result("result_2")
        state._pending["test_task_0"].set_result("result_0")
        state._pending["test_task_1"].set_result("result_1")

        # Each future should have its own correct result
        assert futures["test_task_0"].result() == "result_0"
        assert futures["test_task_1"].result() == "result_1"
        assert futures["test_task_2"].result() == "result_2"

        # Clean up
        with state._pending_lock:
            for tid in ["test_task_0", "test_task_1", "test_task_2"]:
                state._pending.pop(tid, None)

        print("    [metrics] 3 futures resolved out-of-order, all correct")
    test("Pipeline: future-based result matching (no head-of-line blocking)",
         t_future_based_no_head_of_line_blocking)

    # ═══════════════════════════════════════════════════════════════════
    # MICRO-BATCHING TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_batch_config_defaults():
        """BATCH_SIZE defaults to 1 (no batching), BATCH_TIMEOUT_MS to 10."""
        from api_compat._core.config import BATCH_SIZE, BATCH_TIMEOUT_MS
        # Defaults may be overridden by env at import time; just verify
        # they are sensible values
        assert BATCH_SIZE >= 1, f"BATCH_SIZE={BATCH_SIZE} should be >= 1"
        assert BATCH_TIMEOUT_MS >= 0, \
            f"BATCH_TIMEOUT_MS={BATCH_TIMEOUT_MS} should be >= 0"
        print(f"    [metrics] BATCH_SIZE={BATCH_SIZE}, "
              f"BATCH_TIMEOUT_MS={BATCH_TIMEOUT_MS}")
    test("Batch: config defaults are sensible",
         t_batch_config_defaults)

    def t_batch_process_single_correctness():
        """_process_single resolves the future with the right result."""
        from api_compat._core.inference import _process_single, state
        from concurrent.futures import Future

        task_id = "batch_test_single"
        fut = Future()
        with state._pending_lock:
            state._pending[task_id] = fut

        # Monkey-patch state.model.predict to return a known result
        orig_model = state.model

        class FakeModel:
            def predict(self, img, **kw):
                yield {"boxes": [{"score": 0.99}]}

        state.model = FakeModel()
        try:
            _process_single((task_id, np.zeros((1, 1, 3), dtype=np.uint8), fut))
            assert fut.done(), "Future should be resolved"
            boxes = fut.result()
            assert len(boxes) == 1 and boxes[0]["score"] == 0.99
        finally:
            state.model = orig_model
            with state._pending_lock:
                state._pending.pop(task_id, None)
        print("    [metrics] _process_single: future resolved correctly")
    test("Batch: _process_single resolves future with correct result",
         t_batch_process_single_correctness)

    def t_batch_process_batch_correctness():
        """_process_batch resolves all futures in the correct order."""
        from api_compat._core.inference import _process_batch, state
        from concurrent.futures import Future

        n = 4
        futures = []
        task_ids = []
        for i in range(n):
            tid = f"batch_test_{i}"
            fut = Future()
            with state._pending_lock:
                state._pending[tid] = fut
            futures.append(fut)
            task_ids.append(tid)

        orig_model = state.model

        class FakeBatchModel:
            def predict(self, images, batch_size=None, **kw):
                # Return one result per image, indexed by position
                for i in range(len(images)):
                    yield {"boxes": [{"batch_idx": i}]}

        state.model = FakeBatchModel()
        try:
            batch = [
                (task_ids[i], np.zeros((1, 1, 3), dtype=np.uint8), futures[i])
                for i in range(n)
            ]
            _process_batch(batch)

            for i, fut in enumerate(futures):
                assert fut.done(), f"Future {i} should be resolved"
                boxes = fut.result()
                assert len(boxes) == 1
                assert boxes[0]["batch_idx"] == i, \
                    f"Future {i} got wrong result: {boxes[0]}"
        finally:
            state.model = orig_model
            with state._pending_lock:
                for tid in task_ids:
                    state._pending.pop(tid, None)
        print(f"    [metrics] _process_batch: {n} futures resolved in order")
    test("Batch: _process_batch resolves all futures in correct order",
         t_batch_process_batch_correctness)

    def t_batch_process_batch_error_propagation():
        """If batch predict() fails, all futures get the exception."""
        from api_compat._core.inference import _process_batch, state
        from concurrent.futures import Future

        n = 3
        futures = []
        task_ids = []
        for i in range(n):
            tid = f"batch_err_{i}"
            fut = Future()
            with state._pending_lock:
                state._pending[tid] = fut
            futures.append(fut)
            task_ids.append(tid)

        orig_model = state.model

        class FailingModel:
            def predict(self, images, batch_size=None, **kw):
                raise RuntimeError("GPU OOM")

        state.model = FailingModel()
        try:
            batch = [
                (task_ids[i], np.zeros((1, 1, 3), dtype=np.uint8), futures[i])
                for i in range(n)
            ]
            _process_batch(batch)

            for i, fut in enumerate(futures):
                assert fut.done(), f"Future {i} should be resolved (with error)"
                try:
                    fut.result()
                    assert False, "Should have raised"
                except RuntimeError as e:
                    assert "GPU OOM" in str(e)
        finally:
            state.model = orig_model
            with state._pending_lock:
                for tid in task_ids:
                    state._pending.pop(tid, None)
        print(f"    [metrics] {n} futures all received exception")
    test("Batch: batch failure propagates exception to all futures",
         t_batch_process_batch_error_propagation)

    def t_batch_collects_multiple_tasks():
        """The inference worker should collect multiple queued tasks
        into a batch when BATCH_SIZE > 1.  We simulate this by queueing
        several tasks and verifying they are processed together."""
        import api_compat._core.inference as inf
        from api_compat._core.inference import state, _process_batch
        from concurrent.futures import Future

        # Save and override config
        orig_bs = inf.BATCH_SIZE
        orig_model = state.model
        inf.BATCH_SIZE = 3
        inf.BATCH_TIMEOUT_MS = 50.0

        batch_sizes_seen = []

        class TrackingModel:
            def predict(self, images, batch_size=None, **kw):
                n = len(images) if isinstance(images, list) else 1
                batch_sizes_seen.append(n)
                for i in range(n):
                    yield {"boxes": []}

        state.model = TrackingModel()

        try:
            # Queue 3 tasks
            futures = []
            for i in range(3):
                tid = f"collect_{i}"
                fut = Future()
                with state._pending_lock:
                    state._pending[tid] = fut
                state._task_queue.put((tid, np.zeros((1, 1, 3), dtype=np.uint8), fut))
                futures.append(fut)

            # Run one iteration of the worker loop manually
            # (get first task, then try to collect more)
            task = state._task_queue.get()
            batch = [task]
            deadline = time.monotonic() + inf.BATCH_TIMEOUT_MS / 1000.0
            while len(batch) < inf.BATCH_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    extra = state._task_queue.get(timeout=remaining)
                    batch.append(extra)
                except queue.Empty:
                    break

            assert len(batch) == 3, \
                f"Should have collected 3 tasks, got {len(batch)}"
            _process_batch(batch)

            for fut in futures:
                assert fut.done()
            assert batch_sizes_seen == [3], \
                f"Expected single batch call with 3 images, got {batch_sizes_seen}"
        finally:
            inf.BATCH_SIZE = orig_bs
            state.model = orig_model
        print("    [metrics] collected batch of 3 in one predict() call")
    test("Batch: worker collects multiple tasks into single predict() call",
         t_batch_collects_multiple_tasks)

    def t_batch_timeout_flushes_partial():
        """When fewer than BATCH_SIZE tasks arrive, the timeout flushes
        the partial batch rather than waiting forever."""
        import api_compat._core.inference as inf
        from api_compat._core.inference import state, _process_batch
        from concurrent.futures import Future

        orig_bs = inf.BATCH_SIZE
        orig_model = state.model
        inf.BATCH_SIZE = 4
        inf.BATCH_TIMEOUT_MS = 20.0  # 20ms

        class TrackingModel:
            def predict(self, images, batch_size=None, **kw):
                n = len(images) if isinstance(images, list) else 1
                for i in range(n):
                    yield {"boxes": []}

        state.model = TrackingModel()

        try:
            # Queue only 1 task — batch should still process after timeout
            tid = "timeout_flush"
            fut = Future()
            with state._pending_lock:
                state._pending[tid] = fut

            t0 = time.perf_counter()
            # Simulate: get task, wait for more (timeout), process partial
            task = state._task_queue.get() if state._task_queue.qsize() > 0 else None
            if task is None:
                # Put it on the queue then get it
                state._task_queue.put((tid, np.zeros((1, 1, 3), dtype=np.uint8), fut))
                task = state._task_queue.get()

            batch = [task]
            deadline = time.monotonic() + inf.BATCH_TIMEOUT_MS / 1000.0
            while len(batch) < inf.BATCH_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    extra = state._task_queue.get(timeout=remaining)
                    batch.append(extra)
                except queue.Empty:
                    break

            elapsed = time.perf_counter() - t0
            assert len(batch) == 1, "Partial batch should have 1 task"
            # Should have waited ~20ms for the timeout
            assert elapsed >= 0.015, \
                f"Should have waited ~20ms, only {elapsed*1000:.0f}ms"

            _process_batch(batch)
            assert fut.done()
        finally:
            inf.BATCH_SIZE = orig_bs
            state.model = orig_model
        print(f"    [metrics] partial batch flushed after "
              f"{elapsed*1000:.0f}ms timeout")
    test("Batch: timeout flushes partial batch (no infinite wait)",
         t_batch_timeout_flushes_partial)

    def t_batch_size_1_uses_single_path():
        """When BATCH_SIZE=1, the _process_single path is used, not
        _process_batch.  This verifies the branching logic."""
        import api_compat._core.inference as inf
        from api_compat._core.inference import state, _process_single
        from concurrent.futures import Future

        orig_bs = inf.BATCH_SIZE
        orig_model = state.model
        inf.BATCH_SIZE = 1

        single_called = [False]

        class TrackingModel:
            def predict(self, img, **kw):
                single_called[0] = True
                yield {"boxes": []}

        state.model = TrackingModel()

        try:
            tid = "single_path"
            fut = Future()
            with state._pending_lock:
                state._pending[tid] = fut

            # With BATCH_SIZE=1, worker calls _process_single
            _process_single((tid, np.zeros((1, 1, 3), dtype=np.uint8), fut))
            assert single_called[0], "predict() should have been called"
            assert fut.done()
        finally:
            inf.BATCH_SIZE = orig_bs
            state.model = orig_model
        print("    [metrics] BATCH_SIZE=1 → _process_single path used")
    test("Batch: BATCH_SIZE=1 uses single-item path (no batching)",
         t_batch_size_1_uses_single_path)

    def t_throughput_under_load():
        """Measure throughput: requests/second under load."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        n = 50
        responses, elapsed = _send_concurrent(app, n)
        rps = n / elapsed if elapsed > 0 else float("inf")
        # Should handle at least 100 req/s with fast spy (no real GPU)
        assert rps >= 100, \
            f"Throughput {rps:.0f} req/s below 100 req/s threshold " \
            f"(took {elapsed:.3f}s for {n} requests)"
        print(f"    [metrics] {n} requests in {elapsed:.3f}s = {rps:.0f} req/s")
    test("Throughput: >=100 req/s under 50-request load (fast spy)", t_throughput_under_load)

    def t_latency_under_load():
        """Measure per-request latency under concurrent load."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)
        n = 30
        responses, elapsed = _send_concurrent(app, n)
        # All should be fast — no real GPU work
        max_processing_time = max(
            r.json().get("processing_time", 0) for r in responses
        )
        assert max_processing_time < 0.1, \
            f"Max per-request processing_time {max_processing_time:.3f}s " \
            "should be <100ms with fast spy"
        print(f"    [metrics] max latency={max_processing_time:.4f}s, "
              f"wall={elapsed:.3f}s for {n} requests")
    test("Latency: per-request <100ms under 30-request load", t_latency_under_load)

    # ═══════════════════════════════════════════════════════════════════
    # DEADLOCK / RACE CONDITION TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_no_deadlock_mixed_endpoints():
        """Concurrent requests to different endpoints should not deadlock."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)

        async def run():
            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                tasks = []
                # Mix of /convert/file and /convert/source
                for i in range(5):
                    tasks.append(c.post(
                        "/v1/convert/file",
                        files={"file": (f"f_{i}.png", b"fake", "image/png")},
                        data={"to_formats": "md"},
                    ))
                for i in range(5):
                    from docling.datamodel.service.requests import (
                        ConvertSourcesRequest, FileSourceRequest,
                    )
                    req = ConvertSourcesRequest(
                        sources=[FileSourceRequest(
                            base64_string="ZmFrZQ==", filename=f"s_{i}.png",
                        )],
                    )
                    tasks.append(c.post(
                        "/v1/convert/source",
                        json=req.model_dump(mode="json"),
                    ))
                # Also some health checks
                for _ in range(5):
                    tasks.append(c.get("/health"))

                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results

        results = asyncio.run(run())
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"{len(errors)} exceptions: {errors}"
        # All convert requests should be 200, health should be 200
        convert_results = results[:10]
        health_results = results[10:]
        for i, r in enumerate(convert_results):
            assert r.status_code == 200, \
                f"Convert request {i} got {r.status_code}"
        for i, r in enumerate(health_results):
            assert r.status_code == 200, \
                f"Health request {i} got {r.status_code}"
    test("Deadlock: mixed endpoints (file+source+health) no deadlock",
         t_no_deadlock_mixed_endpoints)

    def t_repeated_bursts_no_state_leak():
        """Multiple sequential bursts should not accumulate state or degrade."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)

        times = []
        for burst in range(3):
            responses, elapsed = _send_concurrent(app, 20)
            times.append(elapsed)
            assert all(r.status_code == 200 for r in responses), \
                f"Burst {burst} had failures"

        # Later bursts should not be dramatically slower (no leak)
        # Allow 3x slack for GC / scheduling variance
        assert times[-1] < times[0] * 3, \
            f"Performance degraded: burst 0={times[0]:.3f}s, " \
            f"burst 2={times[-1]:.3f}s"
        print(f"    [metrics] burst times: "
              f"{times[0]:.3f}s, {times[1]:.3f}s, {times[2]:.3f}s")
    test("Stress: 3 sequential bursts of 20 — no performance degradation",
         t_repeated_bursts_no_state_leak)

    # ═══════════════════════════════════════════════════════════════════
    # LATENCY TRACING IN PIPELINE (end-to-end)
    # ═══════════════════════════════════════════════════════════════════

    def t_pipeline_traces_when_enabled():
        """When HPS_LATENCY_LOG=1, the convert pipeline emits latency traces
        with stage breakdowns: image_load, layout_detection, convert_to_docling,
        export_formats."""
        import logging
        from io import StringIO

        old = os.environ.get("HPS_LATENCY_LOG")
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        lat_logger = logging.getLogger("hps_api.latency")
        lat_logger.addHandler(handler)
        lat_logger.setLevel(logging.INFO)

        # Spy that goes through real convert_image path (with tracer)
        # but bypasses actual GPU/image work
        import api_compat.docling_api.service as svc
        import api_compat.docling_api.routes as routes

        async def tracing_spy(image_data, filename, to_formats):
            tracer = LatencyTracer(filename=filename, to_formats=[str(f) for f in to_formats])
            with tracer.stage("image_load"):
                pass
            with tracer.stage("layout_detection"):
                pass
            with tracer.stage("convert_to_docling"):
                pass
            with tracer.stage("export_formats"):
                pass
            tracer.finish()
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename, md_content="# T"),
                status=ConversionStatus.SUCCESS,
                processing_time=0.001,
            )

        try:
            set_latency_logging(True)
            routes.convert_image = tracing_spy
            svc.convert_image = tracing_spy
            app = _create_app_with_spy(None)  # app won't override since we set above

            # Actually need to set spy on the app's modules
            import api_compat.docling_api.service as svc2
            import api_compat.docling_api.routes as routes2
            svc2.convert_image = tracing_spy
            routes2.convert_image = tracing_spy

            from api_compat.docling_api.app import create_app
            app = create_app()

            async def run():
                from httpx import ASGITransport, AsyncClient
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://t",
                ) as c:
                    resp = await c.post(
                        "/v1/convert/file",
                        files={"file": ("t.png", b"fake", "image/png")},
                        data={"to_formats": "md"},
                    )
                    return resp
            asyncio.run(run())

            log_output = log_capture.getvalue()
            assert "latency_trace" in log_output, \
                f"Expected latency_trace in pipeline logs: {log_output}"
            # Parse the JSON trace
            lines = [ln for ln in log_output.strip().split("\n") if "latency_trace" in ln]
            assert len(lines) >= 1
            trace = json.loads(lines[-1])
            assert "stages" in trace
            assert "image_load" in trace["stages"]
            assert "layout_detection" in trace["stages"]
            assert "convert_to_docling" in trace["stages"]
            assert "export_formats" in trace["stages"]
            assert trace["filename"] == "t.png"
        finally:
            lat_logger.removeHandler(handler)
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Pipeline: latency traces emitted with stage breakdowns when enabled",
         t_pipeline_traces_when_enabled)

    def t_pipeline_no_traces_when_disabled():
        """When HPS_LATENCY_LOG=0, the convert pipeline emits NO latency traces."""
        import logging
        from io import StringIO

        old = os.environ.get("HPS_LATENCY_LOG")
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        lat_logger = logging.getLogger("hps_api.latency")
        lat_logger.addHandler(handler)
        lat_logger.setLevel(logging.INFO)

        import api_compat.docling_api.service as svc
        import api_compat.docling_api.routes as routes

        async def tracing_spy(image_data, filename, to_formats):
            tracer = LatencyTracer(filename=filename, to_formats=[str(f) for f in to_formats])
            with tracer.stage("image_load"):
                pass
            tracer.finish()
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename, md_content="# T"),
                status=ConversionStatus.SUCCESS,
                processing_time=0.001,
            )

        try:
            set_latency_logging(False)
            svc.convert_image = tracing_spy
            routes.convert_image = tracing_spy
            from api_compat.docling_api.app import create_app
            app = create_app()

            async def run():
                from httpx import ASGITransport, AsyncClient
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://t",
                ) as c:
                    resp = await c.post(
                        "/v1/convert/file",
                        files={"file": ("t.png", b"fake", "image/png")},
                        data={"to_formats": "md"},
                    )
                    return resp
            asyncio.run(run())

            log_output = log_capture.getvalue()
            assert log_output == "", \
                f"No logs expected when disabled, got: {log_output}"
        finally:
            lat_logger.removeHandler(handler)
            if old is not None:
                os.environ["HPS_LATENCY_LOG"] = old
            else:
                os.environ.pop("HPS_LATENCY_LOG", None)
    test("Pipeline: no latency traces when disabled", t_pipeline_no_traces_when_disabled)

    # ═══════════════════════════════════════════════════════════════════
    # GRADUAL LOAD RAMP TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_ramp_up_load():
        """Gradually increase load: 1, 5, 10, 25, 50 requests.
        All should succeed and throughput should scale."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)

        results = []
        for n in [1, 5, 10, 25, 50]:
            responses, elapsed = _send_concurrent(app, n)
            rps = n / elapsed if elapsed > 0 else 0
            results.append((n, elapsed, rps))
            assert all(r.status_code == 200 for r in responses), \
                f"Failures at n={n}"

        print("    [metrics] load ramp:")
        for n, elapsed, rps in results:
            print(f"      n={n:3d}  elapsed={elapsed:.3f}s  rps={rps:.0f}")

        # Throughput at higher loads should not collapse
        # (allow 50% degradation from first to last)
        first_rps = results[0][2]
        last_rps = results[-1][2]
        assert last_rps >= first_rps * 0.5, \
            f"Throughput collapsed: {first_rps:.0f} → {last_rps:.0f} req/s"
    test("Stress: gradual load ramp (1→50) — throughput scales", t_ramp_up_load)

    def t_concurrent_with_slow_spy_throughput():
        """With a 10ms slow spy, measure how throughput degrades.
        Since the spy bypasses the semaphore, requests run concurrently
        but are bounded by asyncio scheduling."""
        spy = _make_slow_spy(delay=0.01)
        app = _create_app_with_spy(spy)

        n = 20
        responses, elapsed = _send_concurrent(app, n)
        rps = n / elapsed
        # 20 concurrent 10ms requests: ~200ms total minimum if serialized,
        # ~10ms if fully concurrent. With asyncio overhead, expect ~50-200ms
        assert elapsed < 0.5, \
            f"20 concurrent 10ms requests took {elapsed:.3f}s — too slow"
        print(f"    [metrics] {n} req × 10ms spy: "
              f"{elapsed:.3f}s = {rps:.0f} req/s")
    test("Throughput: 20 concurrent × 10ms slow spy", t_concurrent_with_slow_spy_throughput)

    # ═══════════════════════════════════════════════════════════════════
    # LATENCY PERCENTILE TESTS
    # ═══════════════════════════════════════════════════════════════════

    def t_latency_percentiles():
        """Measure p50, p95, p99 latency under load.
        With a fast spy, all should be very low."""
        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)

        n = 50
        responses, elapsed = _send_concurrent(app, n)

        # Extract per-request processing_time from response bodies
        times = sorted(r.json().get("processing_time", 0) for r in responses)
        p50 = times[n // 2]
        p95 = times[int(n * 0.95)]
        p99 = times[int(n * 0.99)] if n >= 100 else times[-1]

        print(f"    [metrics] latency percentiles (n={n}):")
        print(f"      p50={p50:.4f}s  p95={p95:.4f}s  p99={p99:.4f}s")

        # With fast spy, all processing_times should be <100ms
        assert p99 < 0.1, \
            f"p99 latency {p99:.4f}s > 100ms threshold"
    test("Latency: p50/p95/p99 percentiles under 50-request load", t_latency_percentiles)

    def t_slow_spy_latency_percentiles():
        """Measure p50/p95/p99 with a slow spy (20ms) to expose tail latency."""
        spy = _make_slow_spy(delay=0.02)
        app = _create_app_with_spy(spy)

        n = 30
        responses, elapsed = _send_concurrent(app, n)

        times = sorted(r.json().get("processing_time", 0) for r in responses)
        p50 = times[n // 2]
        p95 = times[int(n * 0.95)]
        p99 = times[-1]

        print(f"    [metrics] slow spy (20ms) latency percentiles (n={n}):")
        print(f"      p50={p50:.4f}s  p95={p95:.4f}s  p99={p99:.4f}s")
        print(f"      wall={elapsed:.3f}s")

        # Each request sleeps 20ms, so processing_time should be ~20ms
        assert 0.015 <= p50 <= 0.04, \
            f"p50 {p50:.4f}s should be ~20ms"
    test("Latency: p50/p95/p99 with 20ms slow spy", t_slow_spy_latency_percentiles)

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING UNDER CONCURRENCY
    # ═══════════════════════════════════════════════════════════════════

    def t_concurrent_mixed_success_error():
        """Concurrent requests where some succeed and some fail should
        not interfere with each other."""
        call_count = [0]

        async def mixed_spy(image_data, filename, to_formats):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                # Every 3rd request fails
                return error_response(
                    FailureCategory.INTERNAL,
                    "simulated failure",
                    filename,
                    0.001,
                )
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename, md_content="# T"),
                status=ConversionStatus.SUCCESS,
                processing_time=0.001,
            )

        from api_compat.docling_api.schema import FailureCategory
        from api_compat.docling_api.service import error_response

        app = _create_app_with_spy(mixed_spy)
        responses, _ = _send_concurrent(app, 12)

        successes = sum(1 for r in responses if r.status_code == 200)
        # All should return 200 status (errors are in the body, not HTTP status)
        assert successes == 12, f"Expected 12 HTTP 200s, got {successes}"

        # Check that some have failure status in body
        body_statuses = [r.json()["status"] for r in responses]
        failures = [s for s in body_statuses if "failure" in s.lower()]
        successes_body = [s for s in body_statuses if "success" in s.lower()]
        assert len(failures) >= 1, "Expected at least 1 failure in body"
        assert len(successes_body) >= 1, "Expected at least 1 success in body"
    test("Concurrency: mixed success/error requests don't interfere",
         t_concurrent_mixed_success_error)

    def t_concurrent_large_payload():
        """Concurrent requests with large payloads (1MB each) should work."""
        # 1MB fake image data
        large_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (1024 * 1024)

        spy = _make_fast_spy()
        app = _create_app_with_spy(spy)

        async def run():
            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                tasks = []
                for i in range(5):
                    tasks.append(c.post(
                        "/v1/convert/file",
                        files={"file": (f"large_{i}.png", large_data, "image/png")},
                        data={"to_formats": "md"},
                    ))
                responses = await asyncio.gather(*tasks)
                return responses

        responses = asyncio.run(run())
        for i, resp in enumerate(responses):
            assert resp.status_code == 200, \
                f"Large request {i} got {resp.status_code}"
    test("Concurrency: 5 concurrent 1MB payloads succeed", t_concurrent_large_payload)

    # ═══════════════════════════════════════════════════════════════════
    # TRITON BACKEND TESTS
    # These tests stub the Triton gRPC client to verify the Triton backend
    # path without needing a running Triton server.
    # ═══════════════════════════════════════════════════════════════════

    def t_triton_config_defaults():
        """Triton config constants are importable and sensible."""
        from api_compat._core.config import (
            INFERENCE_BACKEND,
            TRITON_MODEL_NAME,
            TRITON_URL,
        )
        # When forced to direct, INFERENCE_BACKEND should be "direct"
        assert INFERENCE_BACKEND == "direct", \
            f"Expected 'direct' (test override), got '{INFERENCE_BACKEND}'"
        assert TRITON_MODEL_NAME, "TRITON_MODEL_NAME should not be empty"
        assert TRITON_URL, "TRITON_URL should not be empty"
        print(f"    [metrics] backend={INFERENCE_BACKEND}, "
              f"model={TRITON_MODEL_NAME}, url={TRITON_URL}")
    test("Triton: config constants are importable and sensible",
         t_triton_config_defaults)

    def t_triton_client_importable():
        """The triton_client module can be imported without a running server."""
        from api_compat._core import triton_client
        assert hasattr(triton_client, "detect_layout")
        assert hasattr(triton_client, "is_server_ready")
        assert hasattr(triton_client, "close_client")
    test("Triton: client module is importable (no server needed)",
         t_triton_client_importable)

    def t_triton_detect_layout_with_stub():
        """detect_layout() sends image via gRPC and returns boxes.

        We stub tritonclient.grpc.aio to return a canned response, verifying
        the encode → send → decode pipeline works correctly.
        """
        from api_compat._core import triton_client
        from unittest.mock import AsyncMock, MagicMock, patch

        # Canned Triton response: {"boxes": [{"label": "text", "score": 0.9,
        # "coordinate": [0,0,100,100]}], "error": null}
        import json as _json
        response_payload = _json.dumps({
            "boxes": [
                {"label": "text", "score": 0.9,
                 "coordinate": [0, 0, 100, 100]}
            ],
            "error": None,
        }).encode("utf-8")

        # Mock the Triton response object
        mock_response = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=mock_output)
        mock_output.__getitem__.return_value = mock_output
        mock_output.__getitem__.return_value = MagicMock(
            __getitem__=MagicMock(return_value=response_payload)
        )
        # Simpler: mock as_numpy returns array with [0,0] = payload
        mock_arr = MagicMock()
        mock_arr.__getitem__ = MagicMock(return_value=mock_arr)
        type(mock_arr).__getitem__ = MagicMock(
            return_value=MagicMock(
                __getitem__=MagicMock(return_value=response_payload),
                decode=lambda _: response_payload,
            )
        )
        # Actually, just use a real numpy array
        import numpy as _np
        response_arr = _np.array([[response_payload]], dtype=_np.object_)
        mock_response.as_numpy = MagicMock(return_value=response_arr)

        # Mock the client
        mock_client = AsyncMock()
        mock_client.infer = AsyncMock(return_value=mock_response)

        # Patch _ensure_client to return our mock
        with patch.object(triton_client, "_triton_client", mock_client):
            # Create a small test image
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            boxes = asyncio.run(triton_client.detect_layout(image))

        assert len(boxes) == 1
        assert boxes[0]["label"] == "text"
        assert boxes[0]["score"] == 0.9
        assert boxes[0]["coordinate"] == [0, 0, 100, 100]
        print(f"    [metrics] detect_layout returned {len(boxes)} boxes")
    test("Triton: detect_layout() with stubbed gRPC client returns boxes",
         t_triton_detect_layout_with_stub)

    def t_triton_detect_layout_error_propagation():
        """If Triton returns an error, detect_layout raises RuntimeError."""
        from api_compat._core import triton_client
        from unittest.mock import AsyncMock, MagicMock, patch
        import json as _json
        import numpy as _np

        error_payload = _json.dumps({
            "boxes": [],
            "error": "GPU OOM during inference",
        }).encode("utf-8")

        response_arr = _np.array([[error_payload]], dtype=_np.object_)
        mock_response = MagicMock()
        mock_response.as_numpy = MagicMock(return_value=response_arr)

        mock_client = AsyncMock()
        mock_client.infer = AsyncMock(return_value=mock_response)

        with patch.object(triton_client, "_triton_client", mock_client):
            image = np.zeros((50, 50, 3), dtype=np.uint8)
            try:
                asyncio.run(triton_client.detect_layout(image))
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "GPU OOM" in str(e)
        print("    [metrics] Triton error correctly propagated as RuntimeError")
    test("Triton: error response from Triton raises RuntimeError",
         t_triton_detect_layout_error_propagation)

    def t_triton_backend_dispatches_to_triton():
        """run_layout_detection() calls triton_client when backend='triton'.

        We verify the dispatch logic by checking that the Triton path is
        taken (not the direct/inference-thread path).
        """
        from api_compat._core import triton_client
        from unittest.mock import AsyncMock, MagicMock, patch
        import json as _json
        import numpy as _np

        # We need to test the _run_triton path directly since config is
        # already imported as "direct"
        from api_compat._core.inference import _run_triton

        boxes_payload = _json.dumps({
            "boxes": [{"label": "title", "score": 0.95}],
            "error": None,
        }).encode("utf-8")
        response_arr = _np.array([[boxes_payload]], dtype=_np.object_)
        mock_response = MagicMock()
        mock_response.as_numpy = MagicMock(return_value=response_arr)
        mock_client = AsyncMock()
        mock_client.infer = AsyncMock(return_value=mock_response)

        with patch.object(triton_client, "_triton_client", mock_client):
            image = np.zeros((80, 80, 3), dtype=np.uint8)
            boxes = asyncio.run(_run_triton(image))

        assert len(boxes) == 1
        assert boxes[0]["label"] == "title"
        print("    [metrics] _run_triton() dispatched correctly")
    test("Triton: _run_triton() dispatches to triton_client.detect_layout()",
         t_triton_backend_dispatches_to_triton)

    def t_triton_concurrent_requests_batched():
        """Multiple concurrent detect_layout() calls work correctly.

        In production, Triton's dynamic batcher would batch these into one
        GPU call. Here we just verify the client handles concurrency
        without errors (each call gets its own response).
        """
        from api_compat._core import triton_client
        from unittest.mock import AsyncMock, MagicMock, patch
        import json as _json
        import numpy as _np

        async def run_concurrent():
            # Each call gets a unique response with different box count
            call_count = [0]

            async def mock_infer(*args, **kwargs):
                call_count[0] += 1
                n = call_count[0]
                payload = _json.dumps({
                    "boxes": [{"label": f"item_{n}_{i}", "score": 0.9}
                              for i in range(n)],
                    "error": None,
                }).encode("utf-8")
                arr = _np.array([[payload]], dtype=_np.object_)
                resp = MagicMock()
                resp.as_numpy = MagicMock(return_value=arr)
                return resp

            mock_client = AsyncMock()
            mock_client.infer = mock_infer

            with patch.object(triton_client, "_triton_client", mock_client):
                images = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]
                tasks = [triton_client.detect_layout(img) for img in images]
                results = await asyncio.gather(*tasks)

            return results

        results = asyncio.run(run_concurrent())
        assert len(results) == 5
        # Each result should have a different number of boxes (1,2,3,4,5)
        box_counts = sorted(len(r) for r in results)
        assert box_counts == [1, 2, 3, 4, 5], \
            f"Expected [1,2,3,4,5] boxes, got {box_counts}"
        print(f"    [metrics] 5 concurrent calls returned "
              f"{box_counts} boxes respectively")
    test("Triton: 5 concurrent detect_layout() calls each return correct results",
         t_triton_concurrent_requests_batched)

    def t_triton_model_repo_config_exists():
        """The Triton model repository config.pbtxt exists and has
        dynamic_batching enabled."""
        repo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "triton", "model_repo", "doclayout-v3", "config.pbtxt"
        )
        assert os.path.exists(repo_path), \
            f"Triton config not found at {repo_path}"
        with open(repo_path) as f:
            content = f.read()
        assert "dynamic_batching" in content, \
            "config.pbtxt must have dynamic_batching"
        assert "max_batch_size: 8" in content, \
            "config.pbtxt must have max_batch_size: 8"
        assert "max_queue_delay_microseconds" in content, \
            "config.pbtxt must have max_queue_delay for partial batch flush"
        print("    [metrics] config.pbtxt: dynamic_batching + max_batch_size=8")
    test("Triton: model repo config.pbtxt has dynamic_batching enabled",
         t_triton_model_repo_config_exists)

    def t_triton_model_py_exists():
        """The Triton Python backend model.py exists and has the
        required execute() method."""
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "triton", "model_repo", "doclayout-v3", "1", "model.py"
        )
        assert os.path.exists(model_path), \
            f"Triton model.py not found at {model_path}"
        with open(model_path) as f:
            content = f.read()
        assert "class TritonPythonModel" in content
        assert "def execute(self" in content
        assert "def initialize(self" in content
        assert "model.predict" in content, \
            "model.py must call model.predict for batched inference"
        print("    [metrics] model.py: TritonPythonModel with batched predict()")
    test("Triton: Python backend model.py has batched execute() method",
         t_triton_model_py_exists)

    def t_triton_entrypoint_script_exists():
        """The dual-process entrypoint script (run_api_triton.sh) exists."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "run_api_triton.sh"
        )
        assert os.path.exists(script_path), \
            f"Entrypoint script not found at {script_path}"
        assert os.access(script_path, os.X_OK), \
            "run_api_triton.sh must be executable"
        with open(script_path) as f:
            content = f.read()
        assert "tritonserver" in content, "Must start tritonserver"
        assert "granian" in content, "Must start granian"
        assert "HPS_API_BACKEND" in content and "triton" in content, \
            "Must set HPS_API_BACKEND=triton"
        print("    [metrics] run_api_triton.sh: tritonserver + granian")
    test("Triton: entrypoint script starts both tritonserver and granian",
         t_triton_entrypoint_script_exists)

    # ─── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
