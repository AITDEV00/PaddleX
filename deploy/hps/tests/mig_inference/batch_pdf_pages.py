#!/usr/bin/env python3
"""
Batch all rendered PDF page images against the local PaddleX HPS server.

Renders are in ./pdf_pages/ (200 DPI PNGs from Ghostscript). Sends all pages
concurrently to POST /v1/convert/file with to_formats=json, reports wall time,
per-request timing, and per-page box counts. The server has an internal batching
pipeline, so concurrent submission exercises that path.
"""

import argparse
import glob
import io
import json
import os
import ssl
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(HERE, "pdf_pages")
OUT_JSON = os.path.join(HERE, "pdf_batch_result.json")
OUT_DEBUG = os.path.join(HERE, "pdf_batch_bbox")
LOCAL = "http://localhost:8082"
BOUNDARY = "----pdfbatch_boundary_9f4c"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def post(base, filepath, fields, timeout=180):
    fn = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        data = f.read()
    body = io.BytesIO()

    def fld(n, v):
        body.write(
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{n}"\r\n\r\n'
            f"{v}\r\n".encode()
        )

    body.write(
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fn}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    body.write(data)
    body.write(b"\r\n")
    for k, v in fields.items():
        fld(k, v)
    body.write(f"--{BOUNDARY}--\r\n".encode())

    req = urllib.request.Request(
        base + "/v1/convert/file",
        data=body.getvalue(),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
        return resp.read()


def count_boxes(doc):
    jc = (
        doc.get("document", {}).get("json_content", {})
        if isinstance(doc, dict)
        else {}
    )
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            for k in ("texts", "pictures", "tables", "key_value_items", "form_items"):
                if k in o:
                    walk(o[k])
            if o.get("prov") is not None:
                n += 1
            for k in ("texts", "pictures", "tables", "key_value_items", "form_items", "children"):
                if k in o:
                    walk(o[k])
        elif isinstance(o, list):
            for i in o:
                walk(i)

    walk(jc)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=LOCAL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--debug", action="store_true", help="return annotated bbox images")
    args = ap.parse_args()

    pages = sorted(glob.glob(os.path.join(PAGES_DIR, "*.png")))
    if args.limit:
        pages = pages[: args.limit]
    print(f"Base : {args.base}")
    print(f"Pages to process: {len(pages)}")

    outdir = None
    if args.debug:
        outdir = os.path.join(HERE, "pdf_batch_bbox")
        os.makedirs(outdir, exist_ok=True)

    results = [None] * len(pages)

    def worker(i, p):
        t0 = time.time()
        try:
            fields = {"debug": "true"} if args.debug else {"to_formats": "json"}
            raw = post(args.base, p, fields, args.timeout)
            dt = (time.time() - t0) * 1000
            res = {"file": os.path.basename(p), "ok": True, "ms": round(dt, 1)}
            if args.debug:
                out_path = os.path.join(outdir, os.path.basename(p))
                with open(out_path, "wb") as f:
                    f.write(raw)
                res["boxes"] = -1
                res["saved"] = out_path
            else:
                doc = json.loads(raw)
                res["boxes"] = count_boxes(doc)
            results[i] = res
        except Exception as e:  # noqa: BLE001
            results[i] = {
                "file": os.path.basename(p),
                "ok": False,
                "ms": round((time.time() - t0) * 1000, 1),
                "boxes": 0,
                "error": str(e)[:200],
            }

    wall0 = time.time()
    threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(pages)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.time() - wall0) * 1000

    n_ok = sum(1 for r in results if r["ok"])
    n_err = len(results) - n_ok
    ok_times = [r["ms"] for r in results if r["ok"]]

    print(f"\n=== BATCH of {len(pages)} pages (concurrent) ===")
    print(f"  wall-clock : {wall:.0f} ms  ({wall/len(pages):.0f} ms/page avg)")
    print(f"  ok={n_ok}  err={n_err}")
    if ok_times:
        print(
            f"  per-ok ms : min={min(ok_times):.0f} "
            f"mean={sum(ok_times)/len(ok_times):.0f} max={max(ok_times):.0f}"
        )

    for r in results:
        mark = "OK " if r["ok"] else "ERR"
        err = f"  {r.get('error','')}" if not r["ok"] else ""
        print(f"    {r['file']:16s} {mark} {r['ms']:8.0f} ms  boxes={r['boxes']}{err}")

    summary = {
        "pages": len(pages),
        "ok": n_ok,
        "err": n_err,
        "wall_ms": wall,
        "results": results,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved -> {args.out_json}")


if __name__ == "__main__":
    main()