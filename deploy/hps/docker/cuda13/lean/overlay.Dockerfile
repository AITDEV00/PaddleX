# ─────────────────────────────────────────────────────────────────────────────
# LOCAL DEBUG OVERLAY — cu13 (RTX 5090 / sm_120)
# ─────────────────────────────────────────────────────────────────────────────
# Reuses the already-built local `paddlex-hps:layout-cu13-lean-sm120` (CUDA 13
# lean image, which has paddle + models + TensorRT 10.16.1 pinned + TRT runtime
# engine builder baked in) and overlays ONLY the current api_compat source tree that
# contains the debug bbox renderer (`render_debug_image` + `debug` route args).
#
# Purpose: fast local test of the SAME endpoint used on the H200 MIG cluster,
# but on the local RTX 5090 (sm_120), to diff debug bbox output + latency
# between the two GPUs. No full ~9 GB rebuild — only the small api_compat layer
# changes.
#
# Build (from PaddleX repo root):
#   docker build -t paddlex-hps:layout-cu13-debug \
#     -f deploy/hps/docker/cuda13/lean/overlay.Dockerfile .
#
# Run (RTX 5090, engine built on first boot, ~2-3 min):
#   docker run --rm --gpus all \
#     -e HPS_API_BACKEND=direct -e HPS_API_PRECISION=fp16 \
#     -e HPS_API_STARTUP_TIMEOUT=300 \
#     -p 8082:8080 \
#     paddlex-hps:layout-cu13-debug
# ─────────────────────────────────────────────────────────────────────────────

FROM paddlex-hps:layout-cu13-lean-sm120

USER root

# Overlay the current api_compat code (has the debug bbox renderer) + launcher.
COPY deploy/hps/api_compat/ /opt/api_compat/
COPY deploy/hps/scripts/run_api.sh /opt/scripts/run_api.sh
RUN chmod -R a+rX /opt/api_compat /opt/scripts \
    && chmod +x /opt/scripts/run_api.sh

# cu13 lean builds the engine at runtime on the target GPU. Prefer fp16 for
# consistent diff vs the H200 cu12-lean (which used fp16). Give enough time.
ENV HPS_API_BACKEND=direct
ENV HPS_API_PRECISION=fp16
ENV HPS_API_STARTUP_TIMEOUT=300
ENV HPS_API_ENGINE_DIR=/tmp/paddlex-engines

# return to the runner user (default in the base image)
USER runner
WORKDIR /opt

EXPOSE 8080

ENTRYPOINT ["/opt/scripts/run_api.sh"]