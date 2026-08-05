# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — Docker build Makefile
#
# Builds the Docling API images (in deploy/hps/docker) for different GPU
# architectures and variants. Run from anywhere; the recipes cd to the repo
# root (the build context must be the repo root so `paddlex/`, `models/`, and
# `deploy/hps/` paths resolve).
#
# Usage:
#   make help                                     # show this help
#
#   # Blackwell (RTX 5090, sm_120, CUDA 13.0)
#   make blackwell-base                           # NGC base
#   make blackwell-full                           # full API image
#   make blackwell-lean                           # ★ — 7.02 GB (canonical)
#   make blackwell-lean2                          # alias for blackwell-lean
#
#   # Hopper (H200 / H100 / A100 / L40S, sm_90/80/89, CUDA 12.6)
#   make hopper-base
#   make hopper-full
#   make hopper-lean
#
#   # Legacy (Triton-server deployment variant)
#   make legacy
#
#   # Build any (arch, variant) in one shot
#   make build ARCH=hopper VARIANT=lean TAG=my-tag
#   make build ARCH=blackwell VARIANT=lean
#
# To use `docker` instead of `podman`, pass `ENGINE=docker`.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Runtime selection ──────────────────────────────────────────────────────
ENGINE      ?= podman
REPO_ROOT   := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)
DOCKER_DIR  := deploy/hps/docker

# ─── Architecture / variant → Dockerfile & tag mapping ──────────────────────
# Overridable on the command line (e.g. `make build ARCH=hopper VARIANT=lean`).
ARCH     ?= blackwell
VARIANT  ?= lean
TAG      ?=

# Default image tags per architecture + variant.
TAG_blackwell_base   = paddlex-hps-ngc
TAG_blackwell_full   = paddlex-hps-api
TAG_blackwell_lean   = paddlex-hps-api-lean2
TAG_hopper_base      = paddlex-hps-ngc-cuda12
TAG_hopper_full      = paddlex-hps-api-h200
TAG_hopper_lean      = paddlex-hps-api-h200-lean
TAG_legacy_legacy    = paddlex-hps-api-triton

# Resolve the full dockerfile path for a (ARCH, VARIANT) pair.
# special-cases: multistage → full/Dockerfile.multistage ; legacy → legacy/Dockerfile (no subfolder)
# (wrapped in $(strip) because the multi-line $(if) introduces leading whitespace)
dockerfile = $(strip $(if $(filter legacy,$(1)),$(DOCKER_DIR)/legacy/Dockerfile, \
             $(if $(filter multistage,$(2)),$(DOCKER_DIR)/$(1)/full/Dockerfile.multistage, \
             $(DOCKER_DIR)/$(1)/$(2)/Dockerfile)))

# Default tag for the current (ARCH, VARIANT). Overridable via `make ... TAG=foo`.
tag_name = $(TAG_$(1)_$(2))

# ─── Generic build recipe ───────────────────────────────────────────────────
# Runs in the current shell; ARCH/VARIANT come from target-specific variable
# assignments (set below) or from the command line. The image tag is computed
# HERE (inside the recipe) so target-specific ARCH/VARIANT are in scope.
define build-recipe
	@tag="$(if $(TAG),$(TAG),$(if $(call tag_name,$(ARCH),$(VARIANT)),$(call tag_name,$(ARCH),$(VARIANT)),paddlex-hps-$(ARCH)-$(VARIANT)))"; \
	f="$(call dockerfile,$(ARCH),$(VARIANT))"; \
	if test ! -f "$$f"; then \
	  echo "ERROR: no Dockerfile at $$f (ARCH=$(ARCH) VARIANT=$(VARIANT))"; exit 1; \
	fi; \
	echo ">>> Building $(ENGINE) image '$$tag' from $$f"; \
	cd "$(REPO_ROOT)" && $(ENGINE) build -t "$$tag" -f "$$f" .
endef

# Generic target — builds whatever (ARCH, VARIANT) is currently selected.
build:
	$(build-recipe)

# ── Blackwell (RTX 5090, sm_120, CUDA 13.0) ────────────────────────────────
blackwell-base:  ARCH=blackwell
blackwell-base:  VARIANT=base
blackwell-full:  ARCH=blackwell
blackwell-full:  VARIANT=full
blackwell-lean:  ARCH=blackwell
blackwell-lean:  VARIANT=lean
blackwell-lean2: ARCH=blackwell
blackwell-lean2: VARIANT=lean

# ── Hopper (H200 / H100 / A100 / L40S, CUDA 12.6) ──────────────────────────
hopper-base:  ARCH=hopper
hopper-base:  VARIANT=base
hopper-full:  ARCH=hopper
hopper-full:  VARIANT=full
hopper-lean:  ARCH=hopper
hopper-lean:  VARIANT=lean

# ── Legacy (Triton-server variant) ─────────────────────────────────────────
legacy: ARCH=legacy
legacy: VARIANT=legacy

# All named variants just configure (ARCH, VARIANT) then build.
blackwell-base blackwell-full blackwell-lean blackwell-lean2 \
hopper-base hopper-full hopper-lean legacy: build

# ── Help ────────────────────────────────────────────────────────────────────
.PHONY: help build \
        blackwell-base blackwell-full blackwell-lean blackwell-lean2 \
        hopper-base hopper-full hopper-lean legacy

help:
	@echo "PaddleX HPS Docling Docker images — build targets"
	@echo ""
	@echo "  make blackwell-base   build NGC base (RTX 5090, CUDA 13.0)"
	@echo "  make blackwell-full   build full API image (RTX 5090)"
	@echo "  make blackwell-lean   build lean 7.02GB image (RTX 5090)"
	@echo "  make hopper-base      build NGC base (H200, CUDA 12.6)"
	@echo "  make hopper-full      build full API image (H200)"
	@echo "  make hopper-lean      build lean image (H200, engine built on 1st boot)"
	@echo "  make legacy           build Triton-server variant"
	@echo ""
	@echo "  make build ARCH=<> VARIANT=<>  build any (arch,variant)"
	@echo "    ARCH:    blackwell | hopper | legacy"
  @echo "    VARIANT: base | full | lean | legacy | multistage"
	@echo "  Override ENGINE=docker to use docker instead of podman."