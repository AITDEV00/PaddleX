#!/usr/bin/env python
"""Test the Mistral OCR API layer using the official `mistralai` client.

Usage:
    python tests/test_mistral_ocr_client.py [--url http://localhost:8080]

Sends a `document_url` chunk (the only in-doc reference form the server
supports today) to POST /v1/ocr and prints the parsed OCRResponse.
"""
from __future__ import annotations

import argparse

import httpx
from mistralai.client import Mistral
from mistralai.client.models import DocumentURLChunk

DEFAULT_BASE = (
    "https://inference.adeoaiengine.ecouncil.ae/"
    "models/87bc52a5-ceda-4db1-9b00-469ba3c2a1d2/proxy"
)
DEFAULT_TOKEN = "sk-<your api key here>"


def main() -> int:
    ap = argparse.ArgumentParser(description="Test Mistral OCR API")
    ap.add_argument("--url", default=DEFAULT_BASE)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument(
        "--document-url",
        default="http://localhost:9090/business_letter.png",
        help="URL the server will fetch the document image from",
    )
    ap.add_argument("--model", default="PP-DocLayoutV3")
    args = ap.parse_args()

    # Custom httpx.Client with TLS verification disabled (mirrors curl -k)
    # to talk to the self-signed proxy gateway.
    http = httpx.Client(verify=False)
    client = Mistral(
        api_key=args.token,
        server_url=args.url,
        client=http,
    )

    print(f">>> POST {args.url}/v1/ocr  document={args.document_url}")
    print(f"    token={args.token[:10]}... ssl_verify=OFF")
    resp = client.ocr.process(
        model=args.model,
        document=DocumentURLChunk(document_url=args.document_url),
        include_blocks=True,
    )

    print(f"<<< model = {resp.model}")
    print(f"<<< pages_processed = {resp.usage_info.pages_processed}")
    print(f"<<< doc_size_bytes = {resp.usage_info.doc_size_bytes}")

    for page in resp.pages:
        print(f"  page[{page.index}] dims={page.dimensions} "
              f"markdown_len={len(page.markdown or '')}")
        if page.blocks:
            print(f"    blocks={len(page.blocks)}")
            for b in page.blocks[:8]:
                img_id = getattr(b, "image_id", None)
                print(f"      {type(b).__name__:<16} "
                      f"bbox=({b.top_left_x},{b.top_left_y},{b.bottom_right_x},{b.bottom_right_y}) "
                      f"content={b.content!r} image_id={img_id}")
        else:
            print("    (no blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())