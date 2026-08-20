"""Verify confidence-threshold filtering end-to-end.

The official ``mistralai`` client's ``ocr.process`` has a FIXED signature and
cannot send the PaddleX-native ``threshold`` field (server extension). So:
  1. Use the OFFICIAL client to get the baseline per-block confidence (this is
     the "using the client" part — confirms scores are readable).
  2. Use httpx (the client's own HTTP transport) to POST a raw request with a
     high ``threshold`` and confirm low-confidence boxes are DROPPED.

If threshold filtering works, raising the threshold must reduce block count,
and only blocks whose score >= threshold remain.
"""
import os
import sys

import httpx

HPS = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "hps")
sys.path.insert(
    0, os.path.join(HPS, ".venv-cuda13-py310", "lib", "python3.10", "site-packages")
)

from mistralai.client import Mistral  # noqa: E402

MODEL = "PP-DocLayoutV3"
IMAGE_URL = "http://localhost:9090/08_document_scan.jpg"
BASE = "http://localhost:8080"


def client_baseline() -> list[float]:
    """Use the OFFICIAL client to read per-block confidence scores."""
    client = Mistral(server_url=BASE, api_key="hps-test")
    resp = client.ocr.process(
        model=MODEL,
        document={"type": "document_url", "document_url": IMAGE_URL},
        pages="1",
        include_blocks=True,
        confidence_scores_granularity="block",
    )
    scores = []
    for b in resp.pages[0].blocks:
        cs = b.confidence_scores
        scores.append(float(cs.block_type_confidence_score))
    return scores


def raw_ocr_with_threshold(threshold: float) -> list[float]:
    """Send a raw /v1/ocr request with a threshold; return box scores."""
    payload = {
        "model": MODEL,
        "document": {"type": "document_url", "document_url": IMAGE_URL},
        "pages": "1",
        "include_blocks": True,
        "confidence_scores_granularity": "block",
        "threshold": threshold,
        "include_paddlex_metadata": True,
    }
    r = httpx.post(f"{BASE}/v1/ocr", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Native paddlex boxes carry the authoritative per-box score.
    boxes = data["paddlex"]["pages"][0]["boxes"]
    return [float(b["score"]) for b in boxes]


def main() -> None:
    baseline = client_baseline()
    baseline.sort(reverse=True)
    print(f"Official-client baseline: {len(baseline)} blocks")
    print("  scores:", [f"{s:.3f}" for s in baseline])

    print("\nThreshold sweep (raw /v1/ocr, httpx transport):")
    for thr in (0.5, 0.7, 0.8, 0.9, 0.99):
        scores = raw_ocr_with_threshold(thr)
        scores.sort(reverse=True)
        kept = [s for s in scores if s >= thr - 1e-6]
        print(
            f"  threshold={thr:<5} -> {len(scores)} boxes "
            f"(scores >= thr: {len(kept)})"
        )
        if scores:
            print(f"      min={scores[-1]:.3f} max={scores[0]:.3f} "
                  f"scores={[f'{s:.3f}' for s in scores]}")

    # Hard assertion: raising the threshold must not increase the box count.
    c_5 = len(raw_ocr_with_threshold(0.5))
    c_9 = len(raw_ocr_with_threshold(0.9))
    print(f"\nassert: threshold 0.5 -> {c_5} boxes, 0.9 -> {c_9} boxes")
    assert c_9 <= c_5, "threshold did not reduce box count!"
    if c_9 < c_5:
        print("PASS: higher threshold dropped low-confidence boxes")
    else:
        print("NOTE: no boxes dropped (all boxes >= 0.9); try a higher threshold")

    # Every remaining box must meet the threshold.
    high = raw_ocr_with_threshold(0.9)
    assert all(s >= 0.9 - 1e-6 for s in high), "box below threshold survived!"
    print("PASS: every kept box >= threshold")


if __name__ == "__main__":
    main()