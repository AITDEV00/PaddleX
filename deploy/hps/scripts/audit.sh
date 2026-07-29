#!/bin/bash
# audit.sh — systematic code smell detection for api_compat
# Usage: ./audit.sh [package-dir]
# Default: deploy/hps/api_compat relative to repo root
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_DIR="${1:-${HPS_DIR}/api_compat}"

LINT="/tmp/lint-env/bin"
if [ ! -x "$LINT/ruff" ]; then
    echo "Installing lint tools to /tmp/lint-env..."
    python3 -m venv /tmp/lint-env
    /tmp/lint-env/bin/pip install -q ruff pyflakes vulture
fi

echo "══════════════════════════════════════════════════════════"
echo "  CODE SMELL AUDIT — $(date '+%Y-%m-%d %H:%M')"
echo "  Target: $API_DIR"
echo "══════════════════════════════════════════════════════════"
echo ""

# ─── L1: Automated Tools ─────────────────────────────────────
echo "─── L1: PYFLAKES (unused imports, undefined names) ───"
PYFLAKES_OUT=$($LINT/pyflakes "$API_DIR" 2>&1) || true
if [ -z "$PYFLAKES_OUT" ]; then
    echo "  ✅ Clean"
else
    echo "$PYFLAKES_OUT" | sed 's/^/  /'
fi
echo ""

echo "─── L1: VULTURE (dead code, confidence ≥ 80) ───"
VULTURE_OUT=$($LINT/vulture "$API_DIR" --min-confidence 80 2>&1) || true
if [ -z "$VULTURE_OUT" ]; then
    echo "  ✅ Clean"
else
    echo "$VULTURE_OUT" | sed 's/^/  /'
fi
echo ""

echo "─── L1: RUFF (actionable rules only) ───"
RUFF_OUT=$($LINT/ruff check \
    --select F,E9,PLC,PLE,PLR,PLW,B,SIM,RET \
    --ignore PLR0913,PLR2004,SIM117 \
    "$API_DIR" 2>&1) || true
if echo "$RUFF_OUT" | grep -q "All checks passed"; then
    echo "  ✅ Clean"
else
    echo "$RUFF_OUT" | sed 's/^/  /'
fi
echo ""

# ─── L3: Cross-Reference Checks ──────────────────────────────
echo "─── L3: ENUM ↔ DISPATCH TABLE CROSS-CHECK ───"
# Find enum classes
ENUMS=$(grep -rn "class.*enum.Enum" "$API_DIR" --include="*.py" | grep -v "__pycache__")
echo "$ENUMS" | sed 's/^/  Enum: /'
echo ""

echo "─── L3: UPPER-CASE CONSTANTS (verify each is used) ───"
grep -rn "^[A-Z_]\{3,\} =" "$API_DIR" --include="*.py" \
    | grep -v "__pycache__\|coding:\|noqa\|type:" \
    | sed 's/^/  /'
echo ""

echo "─── L3: COMMENTS NAMING SYMBOLS (verify symbol exists) ───"
grep -rn "#.*_[a-z]" "$API_DIR" --include="*.py" \
    | grep -v "__pycache__\|coding:\|noqa\|type:\|pragma\|# ---" \
    | sed 's/^/  /'
echo ""

echo "══════════════════════════════════════════════════════════"
echo "  L1 automated results above. L2 (per-file checklist) and"
echo "  L3 cross-reference verification must be done manually."
echo "  See: docs/code_smell_detection_technique.md"
echo "══════════════════════════════════════════════════════════"
