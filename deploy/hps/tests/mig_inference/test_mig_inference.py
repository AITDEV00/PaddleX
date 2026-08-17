#!/usr/bin/env python3
"""
H200 MIG inference test — debug bbox images + latency (serial vs batch).

Tests the remote PaddleX HPS Docling endpoint via its HTTPS proxy:
  POST /v1/convert/file   (multipart, debug=1)           -> annotated PNG
  POST /v1/convert/file   (multipart, to_formats=json)    -> JSON (for latency)

Usage:
  python3 test_mig_inference.py [--base URL] [--token TOKEN]
                                 [--no-debug] [--no-latency]
                                 [--serial N] [--batch N]
                                 [--limit N]
"""

import argparse
import glob
import io
import os
import ssl
import sys
import threading
import time
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "input")
OUTPUT_DIR = os.path.join(HERE, "bbox_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Local servers (podman/docker with CDI) need no auth; the H200 MIG proxy does.
LOCAL_BASE = "http://localhost:8082"

DEFAULT_BASE = (
    "https://inference.adeoaiengine.ecouncil.ae/"
    "models/87bc52a5-ceda-4db1-9b00-469ba3c2a1d2/proxy"
)
DEFAULT_TOKEN = "sk-UcXlVrMFZy_ZT4R7w35M3tdc7q9rlpZPanjptiwcMs0"

# The proxy uses a self-signed cert (mirrors curl -k).
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def _post_multipart(base, token, filepath, fields, timeout=600, no_auth=False):
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_data = f.read()

    boundary = "----migtest_boundary_7f3a"
    body = io.BytesIO()

    def _field(name, value):
        body.write(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )

    body.write(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    body.write(file_data)
    body.write(b"\r\n")

    for k, v in fields.items():
        _field(k, v)

    body.write(f"--{boundary}--\r\n".encode())

    url = f"{base}/v1/convert/file"
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if not no_auth:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
        ctype = resp.headers.get("Content-Type", "")
        xdebug = resp.headers.get("X-Debug-Filename") or resp.headers.get(
            "X-Debug-Model"
        )
        return resp.read(), ctype, xdebug


def post_debug(base, token, filepath, timeout=120, no_auth=False):
    return _post_multipart(
        base, token, filepath, {"debug": "true"}, timeout, no_auth
    )


def post_convert(base, token, filepath, timeout=120, no_auth=False):
    t0 = time.time()
    try:
        _post_multipart(
            base, token, filepath, {"to_formats": "json"}, timeout, no_auth
        )
        return (time.time() - t0) * 1000, True
    except Exception:  # noqa: BLE001
        return (time.time() - t0) * 1000, False


def test_debug(base, token, limit=None, outdir=OUTPUT_DIR, no_auth=False):
    print("\n=== DEBUG: annotated PNG with bounding boxes ===")
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))
    if limit:
        files = files[:limit]
    n_ok = n_err = 0
    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(outdir, f"{name}_debug.png")
        t0 = time.time()
        try:
            body, ctype, xdebug = post_debug(
                base, token, filepath, no_auth=no_auth
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {name:34s} {time.time()-t0:6.2f}s exception: {e}")
            n_err += 1
            continue
        elapsed = (time.time() - t0) * 1000
        if ctype.startswith("image/png"):
            with open(out_path, "wb") as f:
                f.write(body)
            im = Image.open(io.BytesIO(body))
            n_ok += 1
            print(
                f"  [OK] {name:34s} {im.size} {elapsed:8.1f} ms  "
                f"-> {out_path}  (x-debug={xdebug})"
            )
        else:
            n_err += 1
            print(
                f"  [ERR] {name:34s} ctype={ctype!r} {elapsed:8.1f} ms "
                f"body={body[:200]!r}"
            )
    print(f"  --- debug summary: {n_ok} ok, {n_err} err ---")


def test_latency(base, token, n_serial, n_batch, no_auth=False):
    print("\n=== LATENCY: serial vs concurrent ===")
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))
    if not files:
        print("  No input images.")
        return
    img = files[0]
    name = os.path.basename(img)

    serial = [
        post_convert(base, token, img, no_auth=no_auth)[0] for _ in range(n_serial)
    ]
    print(f"  Image: {name}")
    print(f"  SERIAL  x{n_serial}:")
    print(f"    per-req ms: {[f'{x:.0f}' for x in serial]}")
    print(
        f"    mean={sum(serial)/len(serial):.0f}  min={min(serial):.0f}  "
        f"max={max(serial):.0f}  total={sum(serial):.0f} ms"
    )

    results = [None] * n_batch
    def worker(i):
        results[i] = post_convert(base, token, img, no_auth=no_auth)
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.time() - t0) * 1000
    times = [r[0] for r in results]
    oks = [r[1] for r in results]
    print(f"  BATCH   x{n_batch} concurrent:")
    print(f"    wall-clock={wall:.0f} ms  avg/req={sum(times)/len(times):.0f} ms")
    print(f"    min={min(times):.0f}  max={max(times):.0f}  "
          f"success={sum(oks)}/{n_batch}")

    if serial and n_batch:
        per_serial = sum(serial) / len(serial)
        print(f"\n  Serial avg/req     = {per_serial:.0f} ms")
        print(f"  Batch wall/req     = {wall/n_batch:.0f} ms")
        print(f"  Batch wall (x{n_batch}) = {wall:.0f} ms  vs  "
              f"Serial total (x{n_serial}) = {sum(serial):.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--serial", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    ap.add_argument("--local", action="store_true", help="use local server (no auth)")
    args = ap.parse_args()

    base = args.base or (LOCAL_BASE if args.local else DEFAULT_BASE)
    no_auth = args.local
    print(f"Base : {base}")
    print(f"Token: {'(none)' if no_auth else args.token[:10] + '...'}")

    if not args.no_debug:
        test_debug(
            base, args.token, args.limit, outdir=args.outdir, no_auth=no_auth
        )
    if not args.no_latency:
        test_latency(base, args.token, args.serial, args.batch, no_auth=no_auth)

    print("\nDone.")


if __name__ == "__main__":
    main()