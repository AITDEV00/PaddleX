#!/usr/bin/env bash
#
# Functional test: send different images to /v1/convert/file and verify
# that docling formats (markdown, json, html, text, doctags, doclang) are
# returned correctly.
#
# Usage (inside container):
#   bash /tmp/test_api_functional.sh
#
set -euo pipefail

BASE_URL="http://localhost:8090"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✅ $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; FAILURES=$((FAILURES+1)); }
info() { echo -e "  ${YELLOW}→ $1${NC}"; }

FAILURES=0
mkdir -p /tmp/test_results

echo "============================================"
echo " Functional API Tests"
echo "============================================"
echo ""

# ─── Test 1: Health check ─────────────────────────────────────────────────────
echo "Test 1: Health check"
health=$(curl -sf "${BASE_URL}/health" 2>/dev/null || echo "")
if [[ "$health" == '{"status":"ok"}' ]]; then
  pass "Health endpoint returns ok"
else
  fail "Health endpoint failed: '$health'"
  echo "API not running. Aborting."
  exit 1
fi
echo ""

# ─── Test images ──────────────────────────────────────────────────────────────
declare -A TEST_IMAGES=(
  ["table_paper"]="/tmp/layout-parser-paper-with-table_fp8.png"
  ["book_page"]="/tmp/book_fp8.png"
  ["restaurant_menu"]="/tmp/01_restaurant_menu.jpg"
  ["road_sign"]="/tmp/02_road_sign.jpg"
  ["form_fields"]="/tmp/form_fields.png"
  ["document_scan"]="/tmp/08_document_scan.jpg"
)

# ─── Test 2: Markdown output for each image ───────────────────────────────────
echo "Test 2: Markdown output for each image"
for name in "${!TEST_IMAGES[@]}"; do
  img="${TEST_IMAGES[$name]}"
  if [[ ! -f "$img" ]]; then
    fail "Image not found: $img"
    continue
  fi
  info "Testing $name ($(basename "$img"))"
  resp=$(curl -sf -X POST "${BASE_URL}/v1/convert/file?to_formats=markdown" \
    -F "file=@${img}" 2>/dev/null) || {
    fail "Request failed for $name"
    continue
  }

  # Save raw response
  echo "$resp" > "/tmp/test_results/${name}_md.json"

  # Check status field
  status=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
  if [[ "$status" == "success" ]]; then
    pass "status=success for $name"
  elif [[ "$status" == "partial_success" ]]; then
    pass "status=partial_success for $name (errors present but conversion completed)"
  else
    fail "status='$status' for $name (expected success or partial_success)"
  fi

  # Check md_content is non-empty
  md_len=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('document',{}).get('md_content','') or ''))" 2>/dev/null)
  if [[ "$md_len" -gt 0 ]]; then
    pass "md_content non-empty ($md_len chars) for $name"
  else
    fail "md_content empty for $name"
  fi

  # Check processing_time
  pt=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('processing_time',0))" 2>/dev/null)
  pass "processing_time=${pt}s for $name"

  # Check errors array
  err_count=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('errors',[])))" 2>/dev/null)
  if [[ "$err_count" == "0" ]]; then
    pass "no errors for $name"
  else
    fail "$err_count errors for $name"
  fi
done
echo ""

# ─── Test 3: All output formats ───────────────────────────────────────────────
echo "Test 3: All output formats (markdown,json,html,text,doctags,doclang)"
img="/tmp/layout-parser-paper-with-table_fp8.png"
info "Testing all formats with $(basename "$img")"
resp=$(curl -sf -X POST "${BASE_URL}/v1/convert/file?to_formats=markdown,json,html,text,doctags,doclang" \
  -F "file=@${img}" 2>/dev/null) || {
  fail "Request failed for all-formats test"
  exit 1
}
echo "$resp" > "/tmp/test_results/all_formats.json"

# Check each format field
for field in md_content json_content html_content text_content doctags_content doclang_content; do
  val_len=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
doc = d.get('document', {})
v = doc.get('${field}')
if v is None:
    print(0)
elif isinstance(v, dict):
    # json_content is a DoclingDocument dict
    print(len(json.dumps(v)))
elif isinstance(v, str):
    print(len(v))
else:
    print(len(str(v)))
" 2>/dev/null)
  if [[ "$val_len" -gt 0 ]]; then
    pass "${field} populated (${val_len} chars)"
  else
    fail "${field} empty or null"
  fi
done

# Verify json_content is a valid DoclingDocument
echo ""
echo "Test 3b: Verify json_content is a DoclingDocument"
docling_check=$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
doc = d.get('document', {}).get('json_content', {})
# DoclingDocument must have schema_name and version
has_schema = 'schema_name' in doc
has_version = 'version' in doc
has_texts = 'texts' in doc
print(f'schema_name={has_schema} version={has_version} texts={has_texts}')
if has_schema:
    print(f'  schema_name={doc.get(\"schema_name\")}')
if has_version:
    print(f'  version={doc.get(\"version\")}')
if has_texts:
    print(f'  num_texts={len(doc.get(\"texts\",[]))}')
" 2>/dev/null)
info "$docling_check"
if echo "$docling_check" | grep -q "schema_name=True"; then
  pass "json_content is a valid DoclingDocument"
else
  fail "json_content missing DoclingDocument fields"
fi
echo ""

# ─── Test 4: Error handling ───────────────────────────────────────────────────
echo "Test 4: Error handling"
info "Sending invalid to_formats"
resp_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/v1/convert/file?to_formats=invalid_format" \
  -F "file=@${img}" 2>/dev/null)
if [[ "$resp_code" == "400" ]]; then
  pass "Invalid format returns 400"
else
  fail "Invalid format returns $resp_code (expected 400)"
fi

info "Sending no file"
resp_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/v1/convert/file" 2>/dev/null)
if [[ "$resp_code" == "422" ]]; then
  pass "Missing file returns 422"
else
  fail "Missing file returns $resp_code (expected 422)"
fi
echo ""

# ─── Test 5: Convert/source endpoint (base64) ─────────────────────────────────
echo "Test 5: /v1/convert/source with base64"
info "Encoding image as base64 and sending JSON"
# Write payload to file to avoid shell argument length limits
python3 -c "
import json, base64
with open('/tmp/01_restaurant_menu.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {
    'sources': [{
        'kind': 'file',
        'base64_string': b64,
        'filename': 'restaurant_menu.jpg'
    }],
    'options': {
        'to_formats': ['markdown', 'json']
    }
}
with open('/tmp/source_payload.json', 'w') as f:
    json.dump(payload, f)
"
resp=$(curl -sf -X POST "${BASE_URL}/v1/convert/source" \
  -H "Content-Type: application/json" \
  -d @/tmp/source_payload.json 2>/dev/null) || {
  fail "Request failed for convert/source"
  exit 1
}
echo "$resp" > "/tmp/test_results/source_b64.json"

status=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
md_len=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('document',{}).get('md_content','') or ''))" 2>/dev/null)
if [[ "$status" == "success" || "$status" == "partial_success" ]] && [[ "$md_len" -gt 0 ]]; then
  pass "convert/source base64: status=$status, md_content=$md_len chars"
else
  fail "convert/source base64: status=$status, md_content=$md_len chars"
fi
echo ""

# ─── Test 6: Verify markdown content structure ────────────────────────────────
echo "Test 6: Verify markdown content has layout elements"
for name in table_paper book_page restaurant_menu road_sign form_fields document_scan; do
  result_file="/tmp/test_results/${name}_md.json"
  if [[ ! -f "$result_file" ]]; then
    continue
  fi
  info "Checking $name markdown structure"
  md_content=$(python3 -c "
import sys, json
d = json.load(open('${result_file}'))
md = d.get('document', {}).get('md_content', '') or ''
# Count markdown elements
lines = md.strip().split('\n')
print(f'  total_lines={len(lines)}')
# Check for headings, tables, lists
has_heading = any(l.startswith('#') for l in lines)
has_table = '|' in md
has_list = any(l.startswith('- ') or l.startswith('* ') for l in lines)
print(f'  has_heading={has_heading} has_table={has_table} has_list={has_list}')
# Show first 3 lines
for l in lines[:3]:
    print(f'  | {l[:80]}')
" 2>/dev/null)
  echo "$md_content"
done
echo ""

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "============================================"
if [[ "$FAILURES" -eq 0 ]]; then
  echo -e "${GREEN}ALL TESTS PASSED${NC}"
else
  echo -e "${RED}${FAILURES} FAILURE(S)${NC}"
fi
echo "============================================"
echo "Results saved to /tmp/test_results/"
exit "$FAILURES"
