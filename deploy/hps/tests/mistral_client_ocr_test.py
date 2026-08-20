"""End-to-end test: drive the running HPS Mistral OCR server with the OFFICIAL
``mistralai`` SDK client and verify confidence scores round-trip.

Assumes:
  - Server running on http://localhost:8080 (run_dev_server.sh)
  - Image server on http://localhost:9090 (python3 -m http.server 9090)
  - ``mistralai==2.9.3`` installed in .venv-cuda13-py310
"""
import os
import sys

HPS = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "hps")
sys.path.insert(
    0, os.path.join(HPS, ".venv-cuda13-py310", "lib", "python3.10", "site-packages")
)

from mistralai.client import Mistral  # noqa: E402

MODEL = "PP-DocLayoutV3"
IMAGE_URL = "http://localhost:9090/08_document_scan.jpg"
BASE = "http://localhost:8080"


def show(obj, label: str) -> bool:
    """Return True if obj carries a confidence_scores field with numbers."""
    cs = getattr(obj, "confidence_scores", None)
    if cs is None:
        print(f"  [{label}] NO confidence_scores")
        return False
    btc = getattr(cs, "block_type_confidence_score", None)
    acc = getattr(cs, "average_content_confidence_score", None)
    print(
        f"  [{label}] block_type_confidence_score={btc!r} "
        f"average_content_confidence_score={acc!r}"
    )
    return btc is not None or acc is not None


def main() -> None:
    client = Mistral(server_url=BASE, api_key="hps-test")

    print("=== 1) granularity=block  (expect per-block confidence) ===")
    resp = client.ocr.process(
        model=MODEL,
        document={"type": "document_url", "document_url": IMAGE_URL},
        pages="1",
        include_blocks=True,
        confidence_scores_granularity="block",
    )
    blocks = getattr(resp.pages[0], "blocks", None) or []
    print(f"  pages: {len(resp.pages)}  blocks: {len(blocks)}")
    n = 0
    for b in blocks:
        if show(b, b.content):
            n += 1
    print(f"  blocks WITH confidence: {n}/{len(blocks)}")

    print("\n=== 2) granularity omitted  (expect NO per-block confidence) ===")
    resp2 = client.ocr.process(
        model=MODEL,
        document={"type": "document_url", "document_url": IMAGE_URL},
        pages="1",
        include_blocks=True,
    )
    blocks2 = getattr(resp2.pages[0], "blocks", None) or []
    n2 = 0
    for b in blocks2:
        if show(b, b.content):
            n2 += 1
    print(f"  blocks WITH confidence (should be 0): {n2}/{len(blocks2)}")

    print("\nDONE")


if __name__ == "__main__":
    main()