"""Image loading utilities.

Handles image decoding from bytes, resizing, and URL fetching.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import httpx
import numpy as np

from .config import DEFAULT_TIMEOUT, MAX_IMAGE_DIM

logger = logging.getLogger("hps_api")

# Singleton HTTP client for URL-based image fetching.
# Reusing the client keeps connection pools warm (DNS, TLS, keep-alive)
# and avoids the ~1-3ms overhead of creating a new AsyncClient per call.
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient (lazily created)."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    return _http_client


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load image bytes → numpy array (H, W, 3) BGR uint8.

    Returns **BGR** (not RGB) because PaddleX's ``ReadImage(format="RGB")``
    pre-op does its own ``cv2.cvtColor(BGR2RGB)`` on numpy arrays.  If we
    converted to RGB here, PaddleX would swap it again → double swap →
    net BGR (correct by accident but wastes ~1-2ms per request).

    By returning BGR, PaddleX's ReadImage does exactly one conversion.
    The DoclingDocument converter (which needs RGB) converts lazily via
    :func:`to_rgb` — only when actually needed.

    Uses cv2.imdecode (faster than PIL for JPEG/PNG) and cv2.resize with
    INTER_LINEAR (3-5x faster than PIL LANCZOS with negligible quality
    difference for layout detection inputs).
    """
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    h, w = img.shape[0], img.shape[1]
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        logger.warning("Image resized from (%d,%d) to (%d,%d)", w, h, new_w, new_h)

    return img  # BGR — PaddleX ReadImage will convert to RGB


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR → RGB (lazy, only when needed by the converter)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


async def fetch_image_from_url(url: str, headers: dict[str, Any]) -> bytes:
    """Resolve image bytes from a URL **or** a ``data:`` URI.

    Supports:
      * ``http(s)://`` — downloaded via the pooled httpx.AsyncClient.
      * ``data:[<mediatype>][;base64],<payload>`` — decoded locally (no
        network). This is what LiteLLM's ``/v1/ocr`` produces when a user
        uploads a file (``convert_file_document_to_url_document`` emits a
        base64 data URI), so the backend can consume it without needing a
        file-upload store.

    Validates the bytes are an image before returning them.
    """
    if url.startswith("data:"):
        return _decode_data_uri(url)

    client = _get_http_client()
    resp = await client.get(url, headers=headers, follow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ValueError(
            f"URL returned non-image content-type '{content_type}' — "
            f"expected image/*"
        )

    return resp.content


def _decode_data_uri(url: str) -> bytes:
    """Decode a ``data:`` URI into raw bytes.

    Accepts both base64 (``data:image/png;base64,...``) and percent-encoded
    (``data:image/png,...``) payloads.  Raises ValueError on malformed input
    or non-image media types.
    """
    prefix, comma, payload = url.partition(",")
    if not comma:
        raise ValueError("malformed data URI (missing ',' delimiter)")

    # prefix is "data:<mediatype>;<params>". Strip the leading "data:".
    meta = prefix[5:]
    mediatype = meta.split(";")[0].strip()
    params = meta.split(";")[1:]

    if mediatype and not mediatype.startswith("image/"):
        raise ValueError(
            f"data URI has non-image media type '{mediatype}' — expected image/*"
        )

    is_base64 = any(p.strip() == "base64" for p in params)
    if is_base64:
        import base64

        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid base64 in data URI: {exc}") from exc

    # Percent-encoded payload → URL-decode.
    import urllib.parse

    return urllib.parse.unquote_to_bytes(payload)
