#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# slim_loop.sh — build → measure → run → validate(Docling client) loop for the
# lean Docker images. Used to iterate on image-size reduction while keeping the
# API functional (health 200 + DoclingServiceClient conversion SUCCESS).
#
# Usage:
#   ./deploy/hps/scripts/slim_loop.sh <tag> [dockerfile] [host_port]
#     tag          image tag to build/run (default: paddlex-hps:layout-cu13-lean-sm120)
#     dockerfile   -f arg (default deploy/hps/docker/cuda13/lean/Dockerfile)
#     host_port    host port to publish 8080 -> (default 8081)
#
# Env overrides:
#   SKIP_BUILD=1   skip the docker build (reuse existing <tag> image)
#   SKIP_TEST=1    stop after health check, skip docling client test
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

TAG="${1:-paddlex-hps:layout-cu13-lean-sm120}"
DOCKERFILE="${2:-deploy/hps/docker/cuda13/lean/Dockerfile}"
PORT="${3:-8081}"
CONTAINER="slim-$(echo "$TAG" | tr -cd '[:alnum:]')-test"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TEST_CLIENT=/tmp/slim_docling_client.py

cd "$ROOT"

echo "════════════════════════════════════════════════════════════"
echo "  TAG=$TAG  FILE=$DOCKERFILE  PORT=$PORT  CONTAINER=$CONTAINER"
echo "════════════════════════════════════════════════════════════"

# Cleanup any stale container from a previous run of the same tag.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# ─── BUILD ────────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo ""
  echo ">>> BUILD: $TAG  ($DOCKERFILE)"
  docker build -t "$TAG" -f "$DOCKERFILE" . 2>&1 | tail -25
fi

echo ""
echo ">>> IMAGE SIZE:"
docker image inspect "$TAG" --format '{{.Size}}' \
  | awk '{printf "%.1f GB  (%d bytes)\n", $1/1073741824, $1}'

# ─── RUN ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> RUN container ($CONTAINER) on port $PORT"
docker run -d --name "$CONTAINER" \
  --device nvidia.com/gpu=all \
  --env HPS_API_BACKEND=direct \
  --env HPS_API_PRECISION=fp8 \
  --env HPS_API_STARTUP_TIMEOUT=90 \
  -p "$PORT:8080" \
  "$TAG" >/dev/null
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

echo ""
echo ">>> WAITING for health on http://localhost:$PORT/health-check"
healthy=0
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health-check" 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    echo "    HEALTHY after ~$((i*2))s"
    healthy=1
    break
  fi
  sleep 2
done
if [[ "$healthy" != "1" ]]; then
  echo "    ✗ NOT HEALTHY. Last logs:"
  docker logs --tail 40 "$CONTAINER" 2>&1
  exit 1
fi

# ─── DOCLING CLIENT VALIDATION ────────────────────────────────────────────
if [[ "${SKIP_TEST:-0}" == "1" ]]; then
  echo ""
  echo ">>> SKIP_TEST=1 — stopping after health check."
  exit 0
fi

echo ""
echo ">>> Installing websockets (for DoclingServiceClient) as root ..."
docker exec -u root "$CONTAINER" pip install -q websockets 2>&1 | tail -3

# Write the docling client test that points at the host-published port.
cat > "$TEST_CLIENT" <<PYEOF
import io, sys, base64
from docling.service_client import DoclingServiceClient
from docling.datasources import DocumentStream

url = "http://localhost:$PORT"
client = DoclingServiceClient(url=url, ws_fallback_to_poll=True, job_timeout=120)

# A minimal valid PNG (1x1 red pixel) as the input stream.
png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

source = DocumentStream(name="test_doc.png", stream=io.BytesIO(png))
res = client.convert(source)
print("RESULT_TYPE:", type(res).__name__)
print("STATUS:", res.status)
print("DOC:", res.document)
print("PASS" if str(res.status) == "SUCCESS" else "FAIL")
PYEOF
docker cp "$TEST_CLIENT" "$CONTAINER:/tmp/slim_client.py"

echo ""
echo ">>> RUNNING DoclingServiceClient test ..."
docker exec -u root "$CONTAINER" python3 /tmp/slim_client.py 2>&1 | tail -15

echo ""
echo ">>> DONE — container is still running as '$CONTAINER' (log cleanup on exit)."