#!/usr/bin/env python3
"""Test the OCR /v1/ocr through the LiteLLM gateway proxy.

Sends the local test image as a data: URI (external URL fetch is blocked on the
gateway network) and checks the native behaviors:
- include_native_labels=true -> per-block `label` preserved
- confidence_scores_granularity="block" -> per-block scores
- threshold filtering
"""
import base64
import json
import ssl
import urllib.request
from collections import Counter

GATEWAY = "https://litellm.ecouncil.ae/v1/ocr"
KEY = "sk-05132025"
MODEL = "PP-DocLayoutV3"
IMAGE = "/home/jyao/ADEO/OCR/PaddleX/deploy/hps/tests/mig_inference/input/08_document_scan.jpg"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def post_data(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        GATEWAY, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        return resp.status, json.loads(resp.read())

def data_uri():
    with open(IMAGE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def main():
    uri = data_uri()
    doc = {"type": "image_url", "image_url": uri}

    print("=== 1) include_native_labels=true ===")
    status, d = post_data({
        "model": MODEL, "document": doc, "include_blocks": True,
        "include_native_labels": True,
    })
    print("HTTP", status)
    if status != 200:
        print(json.dumps(d)[:500]); return
    blocks = d["pages"][0]["blocks"]
    print("blocks:", len(blocks))
    print("label/type:", Counter((b.get("label", "<none>"), b.get("type")) for b in blocks))
    print("with label:", sum(1 for b in blocks if "label" in b))

    print("\n=== 2) confidence_scores_granularity=block ===")
    status, d = post_data({
        "model": MODEL, "document": doc, "include_blocks": True,
        "confidence_scores_granularity": "block",
    })
    print("HTTP", status)
    if status != 200:
        print(json.dumps(d)[:500]); return
    blocks = d["pages"][0]["blocks"]
    n = sum(1 for b in blocks if b.get("confidence_scores"))
    print(f"blocks: {len(blocks)} | with confidence: {n}/{len(blocks)}")
    for b in blocks[:4]:
        cs = b.get("confidence_scores", {})
        print(f"  [{b.get('type')}] block_type_confidence_score={cs.get('block_type_confidence_score')}")

    print("\n=== 3) threshold=0.8 (expect fewer boxes) ===")
    status, d = post_data({
        "model": MODEL, "document": doc, "include_blocks": True,
        "threshold": 0.8,
    })
    print("HTTP", status)
    if status != 200:
        print(json.dumps(d)[:500]); return
    print("blocks at threshold 0.8:", len(d["pages"][0]["blocks"]))


if __name__ == "__main__":
    main()