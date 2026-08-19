#!/usr/bin/env python3
"""Threaded static file server for benchmarking (python -m http.server is
single-threaded, which serializes concurrent image fetches)."""
import argparse
import functools
import http.server
import socketserver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=args.dir)
    handler.log_message = lambda *a, **k: None  # silence

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Threaded file server on :{args.port} serving {args.dir} "
          f"({args.threads} worker threads)")
    httpd.daemon_threads = True
    httpd.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())