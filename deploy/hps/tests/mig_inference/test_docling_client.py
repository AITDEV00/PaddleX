#!/usr/bin/env python3
"""
Test the PaddleX HPS endpoint using the official DoclingServiceClient.

Run with the uv venv that has the docling client installed:
    cd deploy/hps
    .venv-cuda13-py310/bin/python tests/.../test_docling_client.py

Converts a file (image or PDF page) against the remote H200 MIG endpoint or
the local cu13 server, then summarizes the returned DoclingDocument.
The remote proxy uses a self-signed cert, so we inject an httpx client with
verify=False into the DoclingServiceClient (the client has no verify param).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "input")
PDF_PAGES = os.path.join(HERE, "pdf_pages")

LOCAL = "http://localhost:8082"
PROXY = (
    "https://inference.adeoaiengine.ecouncil.ae/"
    "models/87bc52a5-ceda-4db1-9b00-469ba3c2a1d2/proxy"
)
TOKEN = "sk-UcXlVrMFZy_ZT4R7w35M3tdc7q9rlpZPanjptiwcMs0"


def make_client(url, api_key, remote):
    from docling.service_client.client import DoclingServiceClient, StatusWatcherKind

    c = DoclingServiceClient(
        url=url,
        api_key=api_key,
        status_watcher=StatusWatcherKind.POLLING,
        ws_fallback_to_poll=True,
        http_read_timeout=600.0,
        job_timeout=600.0,
    )
    if remote:
        # The remote proxy uses a self-signed cert; DoclingServiceClient builds
        # its httpx.Client internally without a verify option, so swap it out.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        c._http_client = httpx.Client(timeout=600.0, verify=False, headers=headers)
    return c


def summarize(res, filename):
    d = res.document
    levels = Counter()
    labels = Counter()
    n_prov = 0
    for item, lvl in d.iterate_items():
        levels[lvl] += 1
        lbl = getattr(item, "label", None)
        if lbl is not None:
            labels[str(lbl)] += 1
        n_prov += len(getattr(item, "prov", []) or [])
    return {
        "file": filename,
        "status": str(res.status),
        "n_tables": len(d.tables),
        "n_pictures": len(d.pictures),
        "levels": dict(levels),
        "labels": dict(labels),
        "prov": n_prov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="use remote H200 MIG proxy")
    ap.add_argument("--local", action="store_true", help="use local cu13 server")
    ap.add_argument("--path", default=None, help="file or PDF to convert")
    ap.add_argument("--pdf-page", type=int, default=None, help="convert a PDF page image by number (1-36)")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--export", default=None, help="also export doc to this format (md/json)")
    args = ap.parse_args()

    if args.local:
        url, api_key, remote = LOCAL, "", False
    else:
        url, api_key, remote = PROXY, TOKEN, True

    c = make_client(url, api_key, remote)
    print(f"Target : {url}\nRemote : {remote}")

    if args.path:
        files = [args.path]
    elif args.pdf_page:
        files = [os.path.join(PDF_PAGES, f"page-{args.pdf_page:03d}.png")]
    else:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))[: args.limit]

    results = []
    for fp in files:
        print(f"\n=== {os.path.basename(fp)} ===")
        try:
            res = c.convert(fp, raises_on_error=False)
            print("  status:", res.status)
            if res.document is None:
                print("  (no document)")
                continue
            s = summarize(res, os.path.basename(fp))
            results.append(s)
            print(f"  levels : {s['levels']}")
            print(f"  labels : {s['labels']}")
            print(f"  tables : {s['n_tables']}  pictures: {s['n_pictures']}  prov: {s['prov']}")
            if args.export:
                if args.export == "md":
                    text = res.document.export_to_markdown()
                elif args.export == "json":
                    text = json.dumps(res.document.export_to_dict(), ensure_ascii=False)
                else:
                    text = "unsupported export"
                print(f"  export ({args.export}) chars: {len(text)}")
                print("  preview:", text[:150].replace("\n", " "))
        except Exception as e:  # noqa: BLE001
            print("  ERROR:", e)
            results.append({"file": os.path.basename(fp), "status": "ERROR", "error": str(e)})

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  saved -> {args.out_json}")


if __name__ == "__main__":
    main()