# ═══════════════════════════════════════════════════════════════════════════════
# PaddleX HPS Docling API — Docker build Makefile
#
# Builds the Docling API images (in deploy/hps/docker) for different CUDA
# versions and variants. Run from anywhere; the recipes cd to the repo root
# (the build context must be the repo root so `paddlex/`, `models/`, and
# `deploy/hps/` paths resolve).
#
# The images are separated by CUDA version (NOT GPU architecture), because the
# container's CUDA runtime libs must be <= the maximum CUDA the host driver
# supports. The lean images build the TensorRT engine at runtime (on first
# boot), so they work across GPU archs within a given CUDA version.
#
# Usage:
#   make help                                     # show this help
#
#   # CUDA 13.0 (needs a host driver >= CUDA 13; RTX 50-series etc.)
#   make cuda13-base                            # NGC base
#   make cuda13-full                            # full API image
#   make cuda13-lean                            # * — 7.02 GB, universal (engine on first boot)
#   make cuda13-lean2                           # alias for cuda13-lean
#
#   # CUDA 12.6 (needs a host driver >= CUDA 12.6; H200/H100/A100/L40S, RTX 40-series)
#   make cuda12-base
#   make cuda12-full
#   make cuda12-lean
#
#   # Legacy (Triton-server deployment variant)
#   make legacy
#
#   # Build any (cuda, variant) in one shot
#   make build CUDA=cuda12 VARIANT=lean TAG=my-tag
#   make build CUDA=cuda13 VARIANT=lean
#
# To use `docker` instead of `podman`, pass `ENGINE=docker`.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Runtime selection ──────────────────────────────────────────────────────
ENGINE      ?= podman
REPO_ROOT   := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)
DOCKER_DIR  := deploy/hps/docker

# ─── CUDA version / variant -> Dockerfile & tag mapping ─────────────────────
# Overridable on the command line (e.g. `make build CUDA=cuda12 VARIANT=lean`).
CUDA     ?= cuda13
VARIANT  ?= lean
TAG      ?=

# Default image tags per CUDA version + variant.
TAG_cuda13_base   = paddlex-hps-ngc
TAG_cuda13_full   = paddlex-hps-api
TAG_cuda13_lean   = paddlex-hps-api-lean2
TAG_cuda12_base   = paddlex-hps-ngc-cuda12
TAG_cuda12_full   = paddlex-hps-api-h200
TAG_cuda12_lean   = paddlex-hps-api-h200-lean
TAG_legacy_legacy = paddlex-hps-api-triton

# Resolve the full dockerfile path for a (CUDA, VARIANT) pair.
# special-cases: multistage -> full/Dockerfile.multistage ; legacy -> legacy/Dockerfile (no subfolder)
# (wrapped in $(strip) because the multi-line $(if) introduces leading whitespace)
dockerfile = $(strip $(if $(filter legacy,$(1)),$(DOCKER_DIR)/legacy/Dockerfile, \
             $(if $(filter multistage,$(2)),$(DOCKER_DIR)/$(1)/full/Dockerfile.multistage, \
             $(DOCKER_DIR)/$(1)/$(2)/Dockerfile)))

# Default tag for the current (CUDA, VARIANT). Overridable via `make ... TAG=foo`.
tag_name = $(TAG_$(1)_$(2))

# ─── Generic build recipe ───────────────────────────────────────────────────
# Runs in the current shell; CUDA/VARIANT come from target-specific variable
# assignments (set below) or from the command line. The image tag is computed
# HERE (inside the recipe) so target-specific CUDA/VARIANT are in scope.
define build-recipe
	@tag="$(if $(TAG),$(TAG),$(if $(call tag_name,$(CUDA),$(VARIANT)),$(call tag_name,$(CUDA),$(VARIANT)),paddlex-hps-$(CUDA)-$(VARIANT)))"; \
	f="$(call dockerfile,$(CUDA),$(VARIANT))"; \
	if test ! -f "$$f"; then \
	  echo "ERROR: no Dockerfile at $$f (CUDA=$(CUDA) VARIANT=$(VARIANT))"; exit 1; \
	fi; \
	echo ">>> Building $(ENGINE) image '$$tag' from $$f"; \
	cd "$(REPO_ROOT)" && $(ENGINE) build -t "$$tag" -f "$$f" .
endef

# Generic target — builds whatever (CUDA, VARIANT) is currently selected.
build:
	$(build-recipe)

# ── CUDA 13 (needs driver >= CUDA 13; RTX 50-series / Blackwell) ────────────
cuda13-base:  CUDA=cuda13
cuda13-base:  VARIANT=base
cuda13-full:  CUDA=cuda13
cuda13-full:  VARIANT=full
cuda13-lean:  CUDA=cuda13
cuda13-lean:  VARIANT=lean
cuda13-lean2: CUDA=cuda13
cuda13-lean2: VARIANT=lean

# ── CUDA 12 (needs driver >= CUDA 12.6; H200/H100/A100/L40S, RTX 40-series) ─
cuda12-base:  CUDA=cuda12
cuda12-base:  VARIANT=base
cuda12-full:  CUDA=cuda12
cuda12-full:  VARIANT=full
cuda12-lean:  CUDA=cuda12
cuda12-lean:  VARIANT=lean

# ── Legacy (Triton-server variant) ─────────────────────────────────────────
legacy: CUDA=legacy
legacy: VARIANT=legacy

# All named variants just configure (CUDA, VARIANT) then build.
cuda13-base cuda13-full cuda13-lean cuda13-lean2 \
cuda12-base cuda12-full cuda12-lean legacy: build

# ── Help ────────────────────────────────────────────────────────────────────
.PHONY: help build \
        cuda13-base cuda13-full cuda13-lean cuda13-lean2 \
        cuda12-base cuda12-full cuda12-lean legacy

help:
	@echo "PaddleX HPS Docling Docker images — build targets"
	@echo ""
	@echo "  make cuda13-base   build NGC base (CUDA 13.0, needs driver >= CUDA 13)"
	@echo "  make cuda13-full   build full API image (CUDA 13.0)"
	@echo "  make cuda13-lean   build universal lean 7.02GB (engine on 1st boot)"
	@echo "  make cuda12-base   build NGC base (CUDA 12.6, H200/H100/A100/L40S/RTX40)"
	@echo "  make cuda12-full   build full API image (CUDA 12.6)"
	@echo "  make cuda12-lean   build lean image (CUDA 12.6, engine on 1st boot)"
	@echo "  make legacy        build Triton-server variant"
	@echo ""
	@echo "  make build CUDA=<> VARIANT=<>  build any (cuda,variant)"
	@echo "    CUDA:    cuda13 | cuda12 | legacy"
	@echo "    VARIANT: base | full | lean | legacy | multistage"
	@echo "  Override ENGINE=docker to use docker instead of podman."
