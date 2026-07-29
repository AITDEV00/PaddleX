#!/usr/bin/env python3
"""Docling API compatibility contract tests.

Verifies that the api_compat package's schema, routes, service, and converter
modules are fully compatible with upstream docling-slim 2.114.0 and
docling-core 2.87.1.

These tests run **without GPU or PaddleX** — they verify protocol contracts,
field names, enum members, and dispatch table completeness, not inference.

Usage:
    cd /home/jyao/ADEO/OCR/PaddleX/deploy/hps
    python3 tests/test_api_compat.py
"""

from __future__ import annotations

import sys
import os

# Ensure the api_compat package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub paddlex so _core.inference imports don't fail (no GPU needed)
if "paddlex" not in sys.modules:
    _stub = type(sys)("paddlex")
    _stub.create_model = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["paddlex"] = _stub


def main() -> int:
    passed = 0
    failed = 0

    def test(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ─── Schema: re-exports match upstream ─────────────────────────────
    from api_compat.docling_api.schema import (
        OutputFormat, QualityGrade, ErrorItem, ConvertDocumentResponse,
        ExportDocumentResponse, ConfidenceScores, ConvertSourcesRequest,
        DoclingComponentType, FailureCategory,
        ConversionStatus, ProfilingItem, ProfilingScope,
    )

    def t_output_format_members():
        expected = {
            'md', 'json', 'yaml', 'html', 'html_split_page', 'text',
            'doctags', 'vtt', 'doclang', 'dclx', 'chunks',
        }
        assert expected == {m.value for m in OutputFormat}, \
            f"OutputFormat mismatch: {expected ^ {m.value for m in OutputFormat}}"
    test("OutputFormat members match upstream (11)", t_output_format_members)

    def t_error_item_fields():
        err = ErrorItem(
            component_type=DoclingComponentType.MODEL,
            module_name="layout",
            error_message="test",
            category=FailureCategory.INTERNAL,
        )
        assert err.error_message == "test"
        assert err.component_type == DoclingComponentType.MODEL
    test("ErrorItem construction with upstream fields", t_error_item_fields)

    def t_convert_document_response():
        resp = ConvertDocumentResponse(
            document=ExportDocumentResponse(filename="t.png"),
            status=ConversionStatus.SUCCESS,
            errors=[],
            processing_time=0.1,
            timings={"layout": ProfilingItem(
                scope=ProfilingScope.DOCUMENT, count=1, times=[0.05],
            )},
            confidence=ConfidenceScores(
                layout_score=0.9, mean_score=0.9,
                mean_grade=QualityGrade.EXCELLENT,
                low_score=0.8, low_grade=QualityGrade.GOOD,
            ),
        )
        assert resp.document.filename == "t.png"
        assert resp.status == ConversionStatus.SUCCESS
    test("ConvertDocumentResponse with all upstream fields", t_convert_document_response)

    # ─── Service: dispatch tables ──────────────────────────────────────
    from api_compat.docling_api.service import (
        _EXPORTERS, _FIELD_MAP, error_response,
    )

    def t_dispatch_tables():
        assert len(_EXPORTERS) == 6, f"Expected 6 exporters, got {len(_EXPORTERS)}"
        assert len(_FIELD_MAP) == 6, f"Expected 6 field mappings, got {len(_FIELD_MAP)}"
        assert set(_EXPORTERS.keys()) == set(_FIELD_MAP.keys()), \
            "Exporter keys don't match field map keys"
    test("_EXPORTERS and _FIELD_MAP have matching 6 entries", t_dispatch_tables)

    def t_error_response():
        resp = error_response(FailureCategory.INTERNAL, "err", "f.png", 0.1)
        assert resp.status == ConversionStatus.FAILURE
        assert resp.errors[0].error_message == "err"
    test("error_response builds valid ConvertDocumentResponse", t_error_response)

    # ─── Routes: _parse_to_formats (list[str] input) ──────────────────
    from api_compat.docling_api.routes import _parse_to_formats

    def t_parse_none():
        assert _parse_to_formats(None) == [OutputFormat.MARKDOWN]
    test("_parse_to_formats(None) defaults to MARKDOWN", t_parse_none)

    def t_parse_empty_list():
        assert _parse_to_formats([]) == [OutputFormat.MARKDOWN]
    test("_parse_to_formats([]) defaults to MARKDOWN", t_parse_empty_list)

    def t_parse_single():
        assert _parse_to_formats(["md"]) == [OutputFormat.MARKDOWN]
    test("_parse_to_formats(['md']) single value", t_parse_single)

    def t_parse_multi():
        result = _parse_to_formats(["md", "json"])
        assert result == [OutputFormat.MARKDOWN, OutputFormat.JSON]
    test("_parse_to_formats(['md', 'json']) multiple values", t_parse_multi)

    def t_parse_csv_in_element():
        # Defensive: a single element with comma-separated values
        result = _parse_to_formats(["md,json"])
        assert result == [OutputFormat.MARKDOWN, OutputFormat.JSON]
    test("_parse_to_formats(['md,json']) CSV within element", t_parse_csv_in_element)

    def t_parse_three_formats():
        result = _parse_to_formats(["md", "json", "text"])
        assert result == [OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT]
    test("_parse_to_formats(['md', 'json', 'text']) three formats", t_parse_three_formats)

    def t_parse_invalid():
        try:
            _parse_to_formats(["md", "invalid"])
            assert False, "Should have raised"
        except Exception as e:
            from fastapi import HTTPException
            assert isinstance(e, HTTPException)
            assert e.status_code == 400
    test("_parse_to_formats(['md', 'invalid']) raises HTTPException 400", t_parse_invalid)

    # ─── Converter: labels and confidence ──────────────────────────────
    from api_compat.docling_api.converter.labels import PADDLEX_TO_DOCLING
    from docling_core.types.doc.labels import DocItemLabel

    def t_label_mapping():
        assert len(PADDLEX_TO_DOCLING) == 25, \
            f"Expected 25 label mappings, got {len(PADDLEX_TO_DOCLING)}"
        for k, v in PADDLEX_TO_DOCLING.items():
            assert isinstance(v, DocItemLabel), f"{k} maps to {v} (not DocItemLabel)"
    test("PADDLEX_TO_DOCLING has 25 valid DocItemLabel mappings", t_label_mapping)

    from api_compat.docling_api.converter.service import PaddleXToDoclingConverter
    conv = PaddleXToDoclingConverter()

    def t_grade_thresholds():
        assert conv._grade_confidence(0.9) == QualityGrade.EXCELLENT
        assert conv._grade_confidence(0.85) == QualityGrade.EXCELLENT
        assert conv._grade_confidence(0.7) == QualityGrade.GOOD
        assert conv._grade_confidence(0.6) == QualityGrade.GOOD
        assert conv._grade_confidence(0.3) == QualityGrade.POOR
    test("QualityGrade thresholds: >=0.85→EXCELLENT, >=0.6→GOOD, else→POOR", t_grade_thresholds)

    def t_compute_confidence():
        boxes = [{"label": "text", "score": 0.9, "coordinate": [0, 0, 10, 10], "order": 0}]
        cs = conv.compute_confidence(boxes)
        assert cs.layout_score == 0.9
        assert cs.mean_grade == QualityGrade.EXCELLENT
    test("compute_confidence produces valid ConfidenceScores", t_compute_confidence)

    def t_sort_reading_order():
        boxes = [
            {"label": "text", "score": 0.9, "coordinate": [0, 0, 10, 10], "order": 1},
            {"label": "title", "score": 0.8, "coordinate": [0, 20, 10, 30], "order": 0},
        ]
        assert conv._sort_by_reading_order(boxes)[0]["label"] == "title"
    test("_sort_by_reading_order respects order field", t_sort_reading_order)

    # ─── Health schema ─────────────────────────────────────────────────
    from api_compat._core.health.schema import HealthCheckResponse, ReadinessResponse

    def t_health_defaults():
        assert HealthCheckResponse().status == "ok"
        assert ReadinessResponse().status == "ok"
    test("Health schema responses default to 'ok'", t_health_defaults)

    # ─── Converter schema: re-exports ──────────────────────────────────
    from api_compat.docling_api.converter.schema import (
        ConfidenceScores as CS2, QualityGrade as QG2,
    )

    def t_converter_schema_reexports():
        assert CS2 is ConfidenceScores, "converter/schema ConfidenceScores is a copy, not re-export"
        assert QG2 is QualityGrade, "converter/schema QualityGrade is a copy, not re-export"
    test("converter/schema.py re-exports upstream types (not copies)", t_converter_schema_reexports)

    # ─── DoclingDocument export methods ────────────────────────────────
    from docling_core.types.doc import DoclingDocument

    def t_export_methods():
        for m in ['export_to_markdown', 'export_to_text', 'export_to_html',
                   'export_to_doctags', 'export_to_doclang']:
            assert hasattr(DoclingDocument, m), f"Missing DoclingDocument.{m}"
    test("All 5 DoclingDocument export methods exist", t_export_methods)

    # ─── ConvertSourcesRequest defaults ────────────────────────────────
    from docling.datamodel.service.requests import FileSourceRequest

    def t_request_defaults():
        req = ConvertSourcesRequest(
            sources=[FileSourceRequest(base64_string="dGVzdA==", filename="t.png")]
        )
        assert req.options is not None, "options should have a default"
        assert req.options.to_formats == [OutputFormat.MARKDOWN]
    test("ConvertSourcesRequest.options defaults to [MARKDOWN]", t_request_defaults)

    # ─── Router registration ───────────────────────────────────────────
    from api_compat.docling_api.routes import router

    def t_router_routes():
        assert len(router.routes) >= 7, \
            f"Expected at least 7 routes, got {len(router.routes)}"
    test("Router has all 7 convert endpoints registered", t_router_routes)

    # ─── HTTP-level tests (httpx ASGITransport) ───────────────────────
    # These tests send real multipart form data through the FastAPI app,
    # exactly like upstream docling-serve tests. They verify that
    # to_formats is correctly received as a list when sent as repeated
    # form fields (the upstream FormDepends pattern).
    import asyncio

    def t_http_to_formats_list():
        """Upstream pattern: data={'to_formats': ['md', 'json']} → both received."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app
            import api_compat.docling_api.service as svc
            import api_compat.docling_api.routes as routes
            from api_compat.docling_api.schema import (
                ConvertDocumentResponse, ExportDocumentResponse, ConversionStatus,
            )

            received = []

            async def spy(image_data, filename, to_formats):
                received.append(list(to_formats))
                return ConvertDocumentResponse(
                    document=ExportDocumentResponse(filename=filename, md_content="# T"),
                    status=ConversionStatus.SUCCESS,
                    processing_time=0.01,
                )

            svc.convert_image = spy
            routes.convert_image = spy

            app = create_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/v1/convert/file",
                    files={"file": ("t.png", b"fake", "image/png")},
                    data={"to_formats": ["md", "json"]},
                )
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            assert len(received[0]) == 2, f"Expected 2 formats, got {len(received[0])}"
            assert received[0] == [OutputFormat.MARKDOWN, OutputFormat.JSON]
        asyncio.run(run())
    test("HTTP: data={'to_formats': ['md','json']} → both received", t_http_to_formats_list)

    def t_http_to_formats_default():
        """No to_formats field → defaults to [MARKDOWN]."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app
            import api_compat.docling_api.service as svc
            import api_compat.docling_api.routes as routes
            from api_compat.docling_api.schema import (
                ConvertDocumentResponse, ExportDocumentResponse, ConversionStatus,
            )

            received = []

            async def spy(image_data, filename, to_formats):
                received.append(list(to_formats))
                return ConvertDocumentResponse(
                    document=ExportDocumentResponse(filename=filename, md_content="# T"),
                    status=ConversionStatus.SUCCESS,
                    processing_time=0.01,
                )

            svc.convert_image = spy
            routes.convert_image = spy

            app = create_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/v1/convert/file",
                    files={"file": ("t.png", b"fake", "image/png")},
                )
            assert resp.status_code == 200
            assert received[0] == [OutputFormat.MARKDOWN]
        asyncio.run(run())
    test("HTTP: no to_formats → defaults to [MARKDOWN]", t_http_to_formats_default)

    def t_http_to_formats_invalid():
        """Invalid format value → 400 error."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app

            app = create_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/v1/convert/file",
                    files={"file": ("t.png", b"fake", "image/png")},
                    data={"to_formats": ["md", "invalid"]},
                )
            assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
            assert "Invalid to_formats" in resp.json()["detail"]
        asyncio.run(run())
    test("HTTP: invalid to_formats → 400", t_http_to_formats_invalid)

    def t_http_to_formats_three():
        """Three formats sent as repeated fields → all three received."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app
            import api_compat.docling_api.service as svc
            import api_compat.docling_api.routes as routes
            from api_compat.docling_api.schema import (
                ConvertDocumentResponse, ExportDocumentResponse, ConversionStatus,
            )

            received = []

            async def spy(image_data, filename, to_formats):
                received.append(list(to_formats))
                return ConvertDocumentResponse(
                    document=ExportDocumentResponse(filename=filename, md_content="# T"),
                    status=ConversionStatus.SUCCESS,
                    processing_time=0.01,
                )

            svc.convert_image = spy
            routes.convert_image = spy

            app = create_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/v1/convert/file",
                    files={"file": ("t.png", b"fake", "image/png")},
                    data={"to_formats": ["md", "json", "text"]},
                )
            assert resp.status_code == 200
            assert len(received[0]) == 3
            assert received[0] == [
                OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT,
            ]
        asyncio.run(run())
    test("HTTP: data={'to_formats': ['md','json','text']} → all 3 received", t_http_to_formats_three)

    def t_http_openapi_to_formats_is_array():
        """OpenAPI schema must declare to_formats as array, not string."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app

            app = create_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.get("/openapi.json")
            schema = resp.json()
            body_ref = schema["paths"]["/v1/convert/file"]["post"]["requestBody"][
                "content"]["multipart/form-data"]["schema"]["$ref"]
            body_schema = schema["components"]["schemas"][body_ref.split("/")[-1]]
            prop = body_schema["properties"]["to_formats"]
            # Must be array type (anyOf with array), NOT string
            any_of = prop.get("anyOf", [])
            types = [item.get("type") for item in any_of]
            assert "array" in types, \
                f"to_formats should be array type, got anyOf={any_of}"
        asyncio.run(run())
    test("HTTP: OpenAPI schema shows to_formats as array type", t_http_openapi_to_formats_is_array)

    # ─── Official client compatibility ────────────────────────────────
    # These tests use the REAL DoclingServiceClient from docling-slim
    # (from docling.service_client import DoclingServiceClient) to verify
    # our endpoint is wire-compatible with the official client SDK.
    # The official client's _form_encode_options() is the authoritative
    # encoding — if our endpoint can decode what it produces, we're
    # fully docling-client-compatible.

    def t_official_client_form_encoding():
        """Official DoclingServiceClient._form_encode_options produces list[str]
        for to_formats, NOT a JSON string. Our endpoint must handle this."""
        from docling.service_client.client import _BaseDoclingServiceClient
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat

        opts = ConvertDocumentsOptions(to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON])
        serialized = opts.model_dump(mode="json", exclude_none=True)
        encoded = _BaseDoclingServiceClient._form_encode_options(serialized)

        # The official client sends to_formats as a list of strings,
        # NOT as a JSON-encoded string. This is the critical behavior.
        assert isinstance(encoded["to_formats"], list), \
            f"Official client should send list, got {type(encoded['to_formats'])}"
        assert encoded["to_formats"] == ["md", "json"], \
            f"Expected ['md', 'json'], got {encoded['to_formats']}"
        # Verify it's NOT a JSON string (the old bug)
        assert not isinstance(encoded["to_formats"], str), \
            "Official client does NOT JSON-encode to_formats — it sends a list"
    test("Official DoclingServiceClient sends to_formats as list[str] (not JSON)", t_official_client_form_encoding)

    def t_official_client_wire_compat():
        """Send data through our endpoint using the EXACT form-encoded payload
        that the official DoclingServiceClient produces."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app
            import api_compat.docling_api.service as svc
            import api_compat.docling_api.routes as routes
            from api_compat.docling_api.schema import (
                ConvertDocumentResponse, ExportDocumentResponse, ConversionStatus,
            )
            from docling.service_client.client import _BaseDoclingServiceClient
            from docling.datamodel.service.options import ConvertDocumentsOptions
            from docling.datamodel.base_models import OutputFormat

            received = []

            async def spy(image_data, filename, to_formats):
                received.append(list(to_formats))
                return ConvertDocumentResponse(
                    document=ExportDocumentResponse(filename=filename, md_content="# T"),
                    status=ConversionStatus.SUCCESS,
                    processing_time=0.01,
                )

            svc.convert_image = spy
            routes.convert_image = spy

            # Use the official client's actual encoding logic
            opts = ConvertDocumentsOptions(
                to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT],
            )
            serialized = opts.model_dump(mode="json", exclude_none=True)
            encoded = _BaseDoclingServiceClient._form_encode_options(serialized)

            app = create_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                resp = await c.post(
                    "/v1/convert/file",
                    files={"file": ("t.png", b"fake", "image/png")},
                    data=encoded,
                )
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            assert len(received[0]) == 3, \
                f"Expected 3 formats from official client encoding, got {len(received[0])}"
            assert received[0] == [
                OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT,
            ], f"Got {received[0]}"
        asyncio.run(run())
    test("Official DoclingServiceClient wire-format → our endpoint receives all formats", t_official_client_wire_compat)

    def t_official_client_raw_multipart():
        """Verify the raw multipart body the official client produces contains
        repeated to_formats= fields (not a single JSON string field)."""
        from docling.service_client.client import _BaseDoclingServiceClient
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat
        import httpx

        opts = ConvertDocumentsOptions(to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON])
        serialized = opts.model_dump(mode="json", exclude_none=True)
        encoded = _BaseDoclingServiceClient._form_encode_options(serialized)

        # Build the actual multipart request the way httpx does
        req = httpx.Request(
            "POST", "http://test/v1/convert/file",
            files={"file": ("t.png", b"fake", "image/png")},
            data=encoded,
        )
        # httpx uses streaming content for multipart — must read() first
        raw_body = req.read().decode("utf-8", errors="replace")

        # The multipart body must contain TWO separate to_formats parts,
        # one per value — NOT a single JSON array string.
        # Parse the multipart body to extract to_formats field values specifically
        lines = raw_body.split("\r\n")
        to_formats_values = []
        for i, line in enumerate(lines):
            if line.startswith('Content-Disposition: form-data; name="to_formats"'):
                # Multipart format: header line, blank line, value, blank line
                if i + 2 < len(lines):
                    to_formats_values.append(lines[i + 2])
        assert len(to_formats_values) == 2, \
            f"Expected 2 to_formats values in multipart body, found {len(to_formats_values)}: {to_formats_values}\n{raw_body}"
        assert set(to_formats_values) == {"md", "json"}, \
            f"Expected md and json, got {to_formats_values}"
        # Must NOT contain a JSON array string like '["md", "json"]'
        assert '["md"' not in raw_body and "[\"md" not in raw_body, \
            "Multipart body should NOT contain JSON array — should be repeated fields"
    test("Official client raw multipart body has repeated to_formats fields (not JSON)", t_official_client_raw_multipart)

    # ═══════════════════════════════════════════════════════════════════
    # COMPREHENSIVE OFFICIAL CLIENT TEST SUITE
    # ═══════════════════════════════════════════════════════════════════
    # These tests exercise the full DoclingServiceClient encoding pipeline
    # across all field types, custom option combinations, extra-field
    # tolerance, all OutputFormats, response parsing, and non-convert
    # endpoints (health, version, async, batch, source).

    # ── Helper: build the official client's form-encoded payload ───────
    def _official_encode(opts):
        """Return the exact form-data dict the official client sends."""
        from docling.service_client.client import _BaseDoclingServiceClient
        serialized = opts.model_dump(mode="json", exclude_none=True)
        return _BaseDoclingServiceClient._form_encode_options(serialized)

    # ── Helper: run an async HTTP request through the ASGI app ─────────
    def _http_post(data=None, files=None, path="/v1/convert/file", spy_fn=None, json_body=None):
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app
            import api_compat.docling_api.service as svc
            import api_compat.docling_api.routes as routes

            if spy_fn:
                svc.convert_image = spy_fn
                routes.convert_image = spy_fn

            app = create_app()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                if json_body is not None:
                    resp = await c.post(path, json=json_body)
                else:
                    resp = await c.post(path, files=files, data=data)
            return resp
        import asyncio
        return asyncio.run(run())

    def _http_get(path):
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app
            app = create_app()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                resp = await c.get(path)
            return resp
        import asyncio
        return asyncio.run(run())

    # ── Helper: spy that captures to_formats and returns success ───────
    def _make_spy(capture_list):
        from api_compat.docling_api.schema import (
            ConvertDocumentResponse, ExportDocumentResponse, ConversionStatus,
        )

        async def spy(image_data, filename, to_formats):
            capture_list.append(list(to_formats))
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(
                    filename=filename, md_content="# T",
                ),
                status=ConversionStatus.SUCCESS,
                processing_time=0.01,
            )
        return spy

    # ───────────────────────────────────────────────────────────────────
    # 1. Full payload acceptance — ALL default options fields
    # ───────────────────────────────────────────────────────────────────
    def t_official_full_default_options():
        """Send ALL 25+ default options fields through our endpoint.
        FastAPI should silently ignore unknown form fields and accept the
        request (status 200), extracting only to_formats."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions()  # all defaults
        encoded = _official_encode(opts)

        # Must contain many fields beyond just to_formats
        assert len(encoded) > 10, \
            f"Default options should have many fields, got {len(encoded)}"

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200, \
            f"Full default options should be accepted, got {resp.status_code}: {resp.text}"
        assert received[0] == [OutputFormat.MARKDOWN], \
            f"Default to_formats should be [md], got {received[0]}"
    test("Official client: full default options (25+ fields) accepted → 200", t_official_full_default_options)

    # ───────────────────────────────────────────────────────────────────
    # 2. Extra fields do NOT cause 422
    # ───────────────────────────────────────────────────────────────────
    def t_official_extra_fields_no_422():
        """The official client sends fields like do_ocr, force_ocr,
        pdf_backend, etc. that our endpoint doesn't declare. FastAPI must
        NOT return 422 Unprocessable Entity for these extra fields."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat

        opts = ConvertDocumentsOptions(
            to_formats=[OutputFormat.MARKDOWN],
            do_ocr=False,
            force_ocr=True,
            ocr_engine="easyocr",
            ocr_lang=["en", "fr", "de"],
            ocr_preset="high",
            pdf_backend="pypdfium2",
            table_mode="fast",
            table_cell_matching=True,
            pipeline="vlm",
            page_range=(1, 5),
            abort_on_error=True,
            do_table_structure=False,
            include_images=True,
            include_page_images=True,
            images_scale=3.0,
            md_page_break_placeholder="---",
            image_export_mode="embedded",
            do_code_enrichment=True,
            do_formula_enrichment=True,
            do_picture_classification=True,
            do_chart_extraction=True,
            do_picture_description=True,
            picture_description_area_threshold=0.1,
        )
        encoded = _official_encode(opts)

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code != 422, \
            f"Extra fields must NOT cause 422, got 422: {resp.text}"
        assert resp.status_code == 200, \
            f"Expected 200, got {resp.status_code}: {resp.text}"
    test("Official client: 20+ extra form fields do NOT trigger 422", t_official_extra_fields_no_422)

    # ───────────────────────────────────────────────────────────────────
    # 3. Field-type encoding: bool fields
    # ───────────────────────────────────────────────────────────────────
    def t_official_bool_encoding():
        """Boolean fields (do_ocr, force_ocr, etc.) are sent as Python
        bools by _form_encode_options. httpx serializes them as 'True'/'False'.
        Our endpoint should ignore them (not 422)."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions(do_ocr=True, force_ocr=False)
        encoded = _official_encode(opts)

        # Bool fields pass through as Python bool, not JSON-encoded
        assert encoded["do_ocr"] is True, \
            f"do_ocr should be bool True, got {encoded['do_ocr']!r}"
        assert encoded["force_ocr"] is False, \
            f"force_ocr should be bool False, got {encoded['force_ocr']!r}"

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200, f"Bool fields should be accepted: {resp.status_code}"
    test("Official client: bool fields (do_ocr=True, force_ocr=False) encoded and accepted", t_official_bool_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 4. Field-type encoding: enum fields
    # ───────────────────────────────────────────────────────────────────
    def t_official_enum_encoding():
        """Enum fields (image_export_mode, pdf_backend, table_mode, pipeline)
        are serialized to their string values by model_dump(mode='json')."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions(
            image_export_mode="placeholder",
            pdf_backend="docling_parse",
            table_mode="accurate",
            pipeline="standard",
        )
        encoded = _official_encode(opts)

        # Enums serialize to their .value strings
        assert encoded["image_export_mode"] == "placeholder"
        assert encoded["pdf_backend"] == "docling_parse"
        assert encoded["table_mode"] == "accurate"
        assert encoded["pipeline"] == "standard"

        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy([]),
        )
        assert resp.status_code == 200
    test("Official client: enum fields serialized to string values and accepted", t_official_enum_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 5. Field-type encoding: list fields (from_formats, ocr_lang)
    # ───────────────────────────────────────────────────────────────────
    def t_official_list_encoding():
        """List fields of primitives (from_formats, ocr_lang, to_formats)
        pass through as Python lists — httpx sends them as repeated fields."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions(ocr_lang=["en", "fr", "de", "zh"])
        encoded = _official_encode(opts)

        # Primitive lists stay as lists (NOT JSON-encoded)
        assert isinstance(encoded["ocr_lang"], list), \
            f"ocr_lang should be list, got {type(encoded['ocr_lang'])}"
        assert encoded["ocr_lang"] == ["en", "fr", "de", "zh"]

        # from_formats is also a list
        assert isinstance(encoded["from_formats"], list)
        assert "pdf" in encoded["from_formats"]

        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy([]),
        )
        assert resp.status_code == 200
    test("Official client: list fields (ocr_lang, from_formats) sent as repeated fields", t_official_list_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 6. Field-type encoding: float fields
    # ───────────────────────────────────────────────────────────────────
    def t_official_float_encoding():
        """Float fields (images_scale, picture_description_area_threshold)
        pass through as Python floats."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions(
            images_scale=3.5,
            picture_description_area_threshold=0.15,
        )
        encoded = _official_encode(opts)

        assert encoded["images_scale"] == 3.5
        assert encoded["picture_description_area_threshold"] == 0.15
        assert isinstance(encoded["images_scale"], float)

        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy([]),
        )
        assert resp.status_code == 200
    test("Official client: float fields (images_scale) encoded and accepted", t_official_float_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 7. Field-type encoding: tuple → list (page_range)
    # ───────────────────────────────────────────────────────────────────
    def t_official_tuple_encoding():
        """Tuple fields (page_range) are serialized to lists by model_dump.
        _form_encode_options keeps them as lists (primitive list)."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions(page_range=(3, 20))
        encoded = _official_encode(opts)

        # page_range is a tuple but model_dump serializes to list
        assert encoded["page_range"] == [3, 20], \
            f"page_range should be [3, 20], got {encoded['page_range']!r}"
        assert isinstance(encoded["page_range"], list)

        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy([]),
        )
        assert resp.status_code == 200
    test("Official client: tuple field (page_range) → list and accepted", t_official_tuple_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 8. Field-type encoding: dict fields (ocr_custom_config)
    # ───────────────────────────────────────────────────────────────────
    def t_official_dict_encoding():
        """Dict fields are JSON-encoded by _form_encode_options (unlike
        primitive lists which pass through). This is the key distinction."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions(
            ocr_custom_config={"lang": "en", "dpi": 300},
        )
        encoded = _official_encode(opts)

        # Dicts are JSON-encoded to strings
        assert isinstance(encoded["ocr_custom_config"], str), \
            f"ocr_custom_config should be JSON string, got {type(encoded['ocr_custom_config'])}"
        import json
        parsed = json.loads(encoded["ocr_custom_config"])
        assert parsed == {"lang": "en", "dpi": 300}

        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy([]),
        )
        assert resp.status_code == 200
    test("Official client: dict field (ocr_custom_config) JSON-encoded and accepted", t_official_dict_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 9. All OutputFormat values accepted through official encoding
    # ───────────────────────────────────────────────────────────────────
    def t_official_all_output_formats():
        """Every valid OutputFormat value should be accepted when sent
        through the official client's encoding pipeline."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat as OF

        # Test each format individually (those our endpoint supports)
        for fmt in [OF.MARKDOWN, OF.JSON, OF.TEXT, OF.HTML, OF.DOCTAGS, OF.DOCLANG]:
            opts = ConvertDocumentsOptions(to_formats=[fmt])
            encoded = _official_encode(opts)

            received = []
            resp = _http_post(
                data=encoded,
                files={"file": ("t.png", b"fake", "image/png")},
                spy_fn=_make_spy(received),
            )
            assert resp.status_code == 200, \
                f"Format {fmt.value} should be accepted, got {resp.status_code}: {resp.text}"
            assert received[0] == [fmt], \
                f"Expected [{fmt}], got {received[0]}"
    test("Official client: all 6 supported OutputFormats accepted individually", t_official_all_output_formats)

    # ───────────────────────────────────────────────────────────────────
    # 10. All OutputFormats combined in one request
    # ───────────────────────────────────────────────────────────────────
    def t_official_all_formats_combined():
        """Send all 6 supported formats in a single request via official
        encoding. All should be received correctly."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat as OF

        all_fmts = [OF.MARKDOWN, OF.JSON, OF.TEXT, OF.HTML, OF.DOCTAGS, OF.DOCLANG]
        opts = ConvertDocumentsOptions(to_formats=all_fmts)
        encoded = _official_encode(opts)

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200, f"Combined formats: {resp.status_code}"
        assert received[0] == all_fmts, \
            f"Expected all 6 formats, got {received[0]}"
    test("Official client: all 6 OutputFormats in single request → all received", t_official_all_formats_combined)

    # ───────────────────────────────────────────────────────────────────
    # 11. Response is parseable by official client's model
    # ───────────────────────────────────────────────────────────────────
    def t_official_response_parseable():
        """Our response JSON must be parseable by the official client's
        ConvertDocumentResponse.model_validate()."""
        from api_compat.docling_api.schema import (
            ConvertDocumentResponse as OurResp,
            ExportDocumentResponse, ConversionStatus,
        )
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        # Build a response our service would produce
        our_resp = OurResp(
            document=ExportDocumentResponse(
                filename="test.png",
                md_content="# Hello",
                json_content=None,
            ),
            status=ConversionStatus.SUCCESS,
            errors=[],
            processing_time=0.05,
            timings={},
        )
        # Serialize to JSON (what the client receives over HTTP)
        json_str = our_resp.model_dump_json()

        # The official client must be able to parse this
        parsed = OfficialResp.model_validate_json(json_str)
        assert parsed.document.filename == "test.png"
        assert parsed.document.md_content == "# Hello"
        assert parsed.status == ConversionStatus.SUCCESS
        assert parsed.processing_time == 0.05
    test("Official client: our response JSON parseable by ConvertDocumentResponse", t_official_response_parseable)

    # ───────────────────────────────────────────────────────────────────
    # 12. Response with errors is parseable
    # ───────────────────────────────────────────────────────────────────
    def t_official_response_with_errors():
        """Response with ErrorItem list must be parseable by official client."""
        from api_compat.docling_api.schema import (
            ConvertDocumentResponse as OurResp, ExportDocumentResponse,
            ConversionStatus, ErrorItem, DoclingComponentType, FailureCategory,
        )
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        our_resp = OurResp(
            document=ExportDocumentResponse(filename="bad.png"),
            status=ConversionStatus.FAILURE,
            errors=[ErrorItem(
                component_type=DoclingComponentType.MODEL,
                module_name="layout",
                error_message="Inference failed",
                category=FailureCategory.INFERENCE_FAILURE,
            )],
            processing_time=0.0,
        )
        json_str = our_resp.model_dump_json()
        parsed = OfficialResp.model_validate_json(json_str)
        assert parsed.status == ConversionStatus.FAILURE
        assert len(parsed.errors) == 1
        assert parsed.errors[0].error_message == "Inference failed"
    test("Official client: failure response with ErrorItem parseable", t_official_response_with_errors)

    # ───────────────────────────────────────────────────────────────────
    # 13. Response with confidence scores is parseable
    # ───────────────────────────────────────────────────────────────────
    def t_official_response_with_confidence():
        """Response with ConfidenceScores must be parseable by official client."""
        from api_compat.docling_api.schema import (
            ConvertDocumentResponse as OurResp, ExportDocumentResponse,
            ConversionStatus, ConfidenceScores, QualityGrade,
        )
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        our_resp = OurResp(
            document=ExportDocumentResponse(filename="t.png", md_content="# T"),
            status=ConversionStatus.SUCCESS,
            processing_time=0.1,
            confidence=ConfidenceScores(
                layout_score=0.92,
                mean_score=0.88,
                mean_grade=QualityGrade.EXCELLENT,
                low_score=0.75,
                low_grade=QualityGrade.GOOD,
            ),
        )
        json_str = our_resp.model_dump_json()
        parsed = OfficialResp.model_validate_json(json_str)
        assert parsed.confidence is not None
        assert parsed.confidence.layout_score == 0.92
        assert parsed.confidence.mean_grade == QualityGrade.EXCELLENT
    test("Official client: response with ConfidenceScores parseable", t_official_response_with_confidence)

    # ───────────────────────────────────────────────────────────────────
    # 14. Health endpoint — official client HealthCheckResponse
    # ───────────────────────────────────────────────────────────────────
    def t_official_health_endpoint():
        """The official client's health() method calls GET /health and
        parses the response with HealthCheckResponse.model_validate_json().
        Our /health endpoint must return compatible JSON."""
        from docling.datamodel.service.responses import HealthCheckResponse

        resp = _http_get("/health")
        assert resp.status_code == 200, f"Health should return 200, got {resp.status_code}"

        # Official client parses this
        parsed = HealthCheckResponse.model_validate_json(resp.text)
        assert parsed.status == "ok"
    test("Official client: GET /health → parseable by HealthCheckResponse", t_official_health_endpoint)

    # ───────────────────────────────────────────────────────────────────
    # 15. Version endpoint — official client version() method
    # ───────────────────────────────────────────────────────────────────
    def t_official_version_endpoint():
        """The official client's version() method calls GET /version and
        returns response.json() as a dict. Our endpoint must return JSON."""
        resp = _http_get("/version")
        assert resp.status_code == 200, f"Version should return 200, got {resp.status_code}"

        data = resp.json()
        assert isinstance(data, dict), f"Version should return dict, got {type(data)}"
        # Our endpoint returns 'version' key
        assert "version" in data, f"Version response should have 'version' key: {data}"
    test("Official client: GET /version → returns JSON dict with version key", t_official_version_endpoint)

    # ───────────────────────────────────────────────────────────────────
    # 16. Async file endpoint returns 200 with TaskStatusResponse
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_file_200():
        """The official client uses /v1/convert/file/async for async
        conversion. Our endpoint accepts the 'files' (plural) field name
        and returns 200 with a TaskStatusResponse.

        Note: The SDK checks ``status_code != 200`` and raises on any other
        code, so we must return 200 (not 202) here.
        The SDK first tries target_type=presigned_url, then falls back to
        inbody on 422.  We test the inbody path here; the presigned_url
        rejection is tested in t_official_async_file_presigned_422."""
        from api_compat.docling_api.service import convert_image
        from api_compat.docling_api.schema import ConversionStatus

        async def fake_convert(image_data, filename, to_formats):
            from api_compat.docling_api.schema import (
                ConvertDocumentResponse, ExportDocumentResponse,
            )
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename),
                status=ConversionStatus.SUCCESS,
                processing_time=0.01,
            )

        resp = _http_post(
            path="/v1/convert/file/async",
            data={"to_formats": ["md"], "target_type": "inbody"},
            files={"files": ("t.png", b"fake", "image/png")},
            spy_fn=fake_convert,
        )
        assert resp.status_code == 200, \
            f"Async file should return 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "task_id" in body, f"Missing task_id in response: {body}"
        assert body["task_type"] == "convert"
        assert body["task_status"] == "success"
    test("Official client: /v1/convert/file/async → 200 TaskStatusResponse", t_official_async_file_200)

    # 16b. Async file: presigned_url target rejected with 422 (SDK fallback trigger)
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_file_presigned_422():
        """The official SDK first sends target_type=presigned_url.
        Our server must reject it with 422 whose detail contains
        'presigned_url' and 'validation error' so the SDK falls back
        to target_type=inbody.  See ``_should_fallback_from_presigned_target``."""
        resp = _http_post(
            path="/v1/convert/file/async",
            data={"to_formats": ["md"], "target_type": "presigned_url"},
            files={"files": ("t.png", b"fake", "image/png")},
        )
        assert resp.status_code == 422, \
            f"presigned_url should return 422, got {resp.status_code}"
        detail = resp.json().get("detail", "")
        assert "presigned_url" in detail.lower(), \
            f"Detail should mention presigned_url: {detail}"
        assert "validation error" in detail.lower(), \
            f"Detail should mention validation error: {detail}"
    test("Official client: /v1/convert/file/async presigned_url → 422 (fallback trigger)", t_official_async_file_presigned_422)

    # ───────────────────────────────────────────────────────────────────
    # 17. Async source endpoint returns 200 with TaskStatusResponse
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_source_200():
        """The official client uses /v1/convert/source/async. Our endpoint
        accepts the same JSON body as /v1/convert/source and returns 200
        with a TaskStatusResponse.
        Uses target=inbody (the SDK fallback path)."""
        from api_compat.docling_api.service import convert_image

        async def fake_convert(image_data, filename, to_formats):
            from api_compat.docling_api.schema import (
                ConversionStatus, ConvertDocumentResponse,
                ExportDocumentResponse,
            )
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename),
                status=ConversionStatus.SUCCESS,
                processing_time=0.01,
            )

        from docling.datamodel.service.requests import (
            ConvertSourcesRequest, FileSourceRequest,
        )
        from docling.datamodel.service.targets import InBodyTarget

        req = ConvertSourcesRequest(
            sources=[FileSourceRequest(base64_string="dGVzdA==", filename="t.png")],
            target=InBodyTarget(),
        )
        resp = _http_post(
            path="/v1/convert/source/async",
            json_body=req.model_dump(mode="json"),
            spy_fn=fake_convert,
        )
        assert resp.status_code == 200, \
            f"Async source should return 200, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "task_id" in body, f"Missing task_id in response: {body}"
        assert body["task_type"] == "convert"
    test("Official client: /v1/convert/source/async → 200 TaskStatusResponse", t_official_async_source_200)

    # 17b. Async source: presigned_url target rejected with 422 (SDK fallback trigger)
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_source_presigned_422():
        """Same as t_official_async_file_presigned_422 but for the JSON
        source/async endpoint. The target is in the request body as
        ``{"kind": "presigned_url"}``."""
        from docling.datamodel.service.requests import (
            ConvertSourcesRequest, FileSourceRequest,
        )

        req = ConvertSourcesRequest(
            sources=[FileSourceRequest(base64_string="dGVzdA==", filename="t.png")],
        )
        # Override target to presigned_url (default is InBodyTarget)
        body = req.model_dump(mode="json")
        body["target"] = {"kind": "presigned_url"}
        resp = _http_post(
            path="/v1/convert/source/async",
            json_body=body,
        )
        assert resp.status_code == 422, \
            f"presigned_url source should return 422, got {resp.status_code}"
        detail = resp.json().get("detail", "")
        assert "presigned_url" in detail.lower(), \
            f"Detail should mention presigned_url: {detail}"
        assert "validation error" in detail.lower(), \
            f"Detail should mention validation error: {detail}"
    test("Official client: /v1/convert/source/async presigned_url → 422 (fallback trigger)", t_official_async_source_presigned_422)

    # ───────────────────────────────────────────────────────────────────
    # 18. Batch endpoint returns 501
    # ───────────────────────────────────────────────────────────────────
    def t_official_batch_501():
        """The official client uses /v1/convert/source/batch for batch
        conversion. Our endpoint returns 501 Not Implemented.
        BatchConvertSourcesRequest requires cloud sources (http/s3/etc)
        and a target — file sources are not accepted by the batch schema."""
        # Build a valid batch body with an HTTP source and presigned URL target
        batch_body = {
            "sources": [{
                "url": "http://example.com/doc.png",
                "kind": "http",
            }],
            "target": {"kind": "presigned_url"},
        }
        resp = _http_post(
            path="/v1/convert/source/batch",
            json_body=batch_body,
        )
        assert resp.status_code == 501, \
            f"Batch should return 501, got {resp.status_code}"
    test("Official client: /v1/convert/source/batch → 501 Not Implemented", t_official_batch_501)

    # ───────────────────────────────────────────────────────────────────
    # 18b. Async flow: poll task status after submit
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_poll_status():
        """After submitting an async task, the client polls
        /v1/status/poll/{task_id}. The status should reflect the
        conversion result (success or failure)."""

        async def fake_convert(image_data, filename, to_formats):
            from api_compat.docling_api.schema import (
                ConversionStatus, ConvertDocumentResponse,
                ExportDocumentResponse,
            )
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(filename=filename),
                status=ConversionStatus.SUCCESS,
                processing_time=0.01,
            )

        submit_resp = _http_post(
            path="/v1/convert/file/async",
            data={"to_formats": ["md"], "target_type": "inbody"},
            files={"files": ("t.png", b"fake", "image/png")},
            spy_fn=fake_convert,
        )
        assert submit_resp.status_code == 200
        task_id = submit_resp.json()["task_id"]

        poll_resp = _http_get(f"/v1/status/poll/{task_id}")
        assert poll_resp.status_code == 200, \
            f"Poll should return 200, got {poll_resp.status_code}"
        body = poll_resp.json()
        assert body["task_id"] == task_id
        assert body["task_status"] == "success"
    test("Official client: /v1/status/poll/{task_id} → 200 TaskStatusResponse", t_official_async_poll_status)

    # ───────────────────────────────────────────────────────────────────
    # 18c. Async flow: retrieve task result after submit
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_get_result():
        """After polling shows success, the client fetches the result
        from /v1/result/{task_id}. The response must be a
        ConvertDocumentResponse with document data."""

        async def fake_convert(image_data, filename, to_formats):
            from api_compat.docling_api.schema import (
                ConversionStatus, ConvertDocumentResponse,
                ExportDocumentResponse,
            )
            return ConvertDocumentResponse(
                document=ExportDocumentResponse(
                    filename=filename,
                    md_content="# Test",
                ),
                status=ConversionStatus.SUCCESS,
                processing_time=0.05,
            )

        submit_resp = _http_post(
            path="/v1/convert/file/async",
            data={"to_formats": ["md"], "target_type": "inbody"},
            files={"files": ("t.png", b"fake", "image/png")},
            spy_fn=fake_convert,
        )
        assert submit_resp.status_code == 200
        task_id = submit_resp.json()["task_id"]

        result_resp = _http_get(f"/v1/result/{task_id}")
        assert result_resp.status_code == 200, \
            f"Result should return 200, got {result_resp.status_code}"
        body = result_resp.json()
        assert body["status"] == "success"
        assert body["document"]["filename"] == "t.png"
        assert body["document"]["md_content"] == "# Test"
    test("Official client: /v1/result/{task_id} → 200 ConvertDocumentResponse", t_official_async_get_result)

    # ───────────────────────────────────────────────────────────────────
    # 18d. Async flow: poll unknown task returns 404
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_poll_unknown_404():
        """Polling an unknown task_id should return 404."""
        resp = _http_get("/v1/status/poll/nonexistent-task-id")
        assert resp.status_code == 404, \
            f"Unknown task poll should return 404, got {resp.status_code}"
    test("Official client: /v1/status/poll/{unknown} → 404", t_official_async_poll_unknown_404)

    # ───────────────────────────────────────────────────────────────────
    # 18e. Async flow: result for unknown task returns 404
    # ───────────────────────────────────────────────────────────────────
    def t_official_async_result_unknown_404():
        """Retrieving result for unknown task_id should return 404."""
        resp = _http_get("/v1/result/nonexistent-task-id")
        assert resp.status_code == 404, \
            f"Unknown task result should return 404, got {resp.status_code}"
    test("Official client: /v1/result/{unknown} → 404", t_official_async_result_unknown_404)

    # ───────────────────────────────────────────────────────────────────
    # 19. Convert source endpoint (JSON body) — file source
    # ───────────────────────────────────────────────────────────────────
    def t_official_convert_source_file():
        """The /v1/convert/source endpoint accepts a JSON body with
        ConvertSourcesRequest. Test with a base64 file source."""
        from docling.datamodel.service.requests import (
            ConvertSourcesRequest, FileSourceRequest,
        )
        from docling.datamodel.base_models import OutputFormat

        # base64 of "fake" = "ZmFrZQ=="
        req = ConvertSourcesRequest(
            sources=[FileSourceRequest(base64_string="ZmFrZQ==", filename="t.png")],
        )
        received = []
        resp = _http_post(
            path="/v1/convert/source",
            json_body=req.model_dump(mode="json"),
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200, \
            f"Convert source should return 200, got {resp.status_code}: {resp.text}"
        assert received[0] == [OutputFormat.MARKDOWN]
    test("Official client: /v1/convert/source with file source → 200", t_official_convert_source_file)

    # ───────────────────────────────────────────────────────────────────
    # 20. Convert source with custom to_formats
    # ───────────────────────────────────────────────────────────────────
    def t_official_convert_source_custom_formats():
        """ConvertSourcesRequest with custom options.to_formats."""
        from docling.datamodel.service.requests import (
            ConvertSourcesRequest, FileSourceRequest,
        )
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat

        req = ConvertSourcesRequest(
            sources=[FileSourceRequest(base64_string="ZmFrZQ==", filename="t.png")],
            options=ConvertDocumentsOptions(
                to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT],
            ),
        )
        received = []
        resp = _http_post(
            path="/v1/convert/source",
            json_body=req.model_dump(mode="json"),
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        assert received[0] == [OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT]
    test("Official client: /v1/convert/source with custom to_formats → 3 formats received", t_official_convert_source_custom_formats)

    # ───────────────────────────────────────────────────────────────────
    # 21. Multipart body has correct field count for multiple to_formats
    # ───────────────────────────────────────────────────────────────────
    def t_official_multipart_field_count():
        """When official client sends 3 to_formats, the multipart body
        should contain exactly 3 to_formats parts."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat
        import httpx

        opts = ConvertDocumentsOptions(
            to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT],
        )
        encoded = _official_encode(opts)

        req = httpx.Request(
            "POST", "http://test/v1/convert/file",
            files={"file": ("t.png", b"fake", "image/png")},
            data=encoded,
        )
        raw_body = req.read().decode("utf-8", errors="replace")

        # Count to_formats parts
        count = raw_body.count('name="to_formats"')
        assert count == 3, \
            f"Expected 3 to_formats parts, found {count}"

        # Also verify the values are correct
        lines = raw_body.split("\r\n")
        values = []
        for i, line in enumerate(lines):
            if line.startswith('Content-Disposition: form-data; name="to_formats"'):
                if i + 2 < len(lines):
                    values.append(lines[i + 2])
        assert set(values) == {"md", "json", "text"}, f"Got {values}"
    test("Official client: 3 to_formats → exactly 3 multipart parts with correct values", t_official_multipart_field_count)

    # ───────────────────────────────────────────────────────────────────
    # 22. Multipart body includes all option fields
    # ───────────────────────────────────────────────────────────────────
    def t_official_multipart_all_fields_present():
        """When full options are sent, the multipart body should contain
        parts for every field that _form_encode_options produces."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        import httpx

        opts = ConvertDocumentsOptions(
            do_ocr=False, force_ocr=True, ocr_lang=["en", "fr"],
            pdf_backend="docling_parse", table_mode="fast",
            images_scale=2.0, page_range=(1, 5),
        )
        encoded = _official_encode(opts)

        req = httpx.Request(
            "POST", "http://test/v1/convert/file",
            files={"file": ("t.png", b"fake", "image/png")},
            data=encoded,
        )
        raw_body = req.read().decode("utf-8", errors="replace")

        # Every key in encoded should appear as a form field in the body
        for key in encoded:
            assert f'name="{key}"' in raw_body, \
                f"Field '{key}' not found in multipart body"
    test("Official client: all option fields present as multipart parts", t_official_multipart_all_fields_present)

    # ───────────────────────────────────────────────────────────────────
    # 23. Empty to_formats list → default MARKDOWN
    # ───────────────────────────────────────────────────────────────────
    def t_official_empty_to_formats():
        """When official client sends empty to_formats (via exclude_none),
        our endpoint should default to [MARKDOWN]."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        # With exclude_none, to_formats (which has a default) is included
        # But if we explicitly set to_formats=[], exclude_none keeps it
        opts = ConvertDocumentsOptions()
        encoded = _official_encode(opts)

        # Default ConvertDocumentsOptions includes to_formats=[MARKDOWN]
        assert "to_formats" in encoded
        assert encoded["to_formats"] == ["md"]

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200
        assert received[0] == [OutputFormat.MARKDOWN]
    test("Official client: default options → to_formats defaults to [md]", t_official_empty_to_formats)

    # ───────────────────────────────────────────────────────────────────
    # 24. Nested model fields (picture_description_api) are JSON-encoded
    # ───────────────────────────────────────────────────────────────────
    def t_official_nested_model_encoding():
        """Nested model fields like picture_description_api are dicts after
        model_dump, so _form_encode_options JSON-encodes them."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        # picture_description_api expects a dict in model_dump mode,
        # so we pass it as a dict to avoid type validation issues
        opts = ConvertDocumentsOptions(
            do_picture_description=True,
            picture_description_api={
                "url": "http://vlm-service/v1/chat",
                "params": {"model": "test"},
                "timeout": 30,
                "prop_cls": "<picture>",
            },
        )
        encoded = _official_encode(opts)

        # Nested model → dict → JSON-encoded by _form_encode_options
        assert "picture_description_api" in encoded
        assert isinstance(encoded["picture_description_api"], str), \
            f"Nested model should be JSON string, got {type(encoded['picture_description_api'])}"
        import json
        parsed = json.loads(encoded["picture_description_api"])
        assert isinstance(parsed, dict)
        assert parsed.get("url") == "http://vlm-service/v1/chat"

        # Our endpoint should accept this without 422
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy([]),
        )
        assert resp.status_code == 200, \
            f"Nested model field should be accepted: {resp.status_code}"
    test("Official client: nested model fields JSON-encoded and accepted", t_official_nested_model_encoding)

    # ───────────────────────────────────────────────────────────────────
    # 25. OpenAPI schema declares all expected routes
    # ───────────────────────────────────────────────────────────────────
    def t_official_openapi_routes():
        """The OpenAPI schema must declare all routes the official client
        might call: /v1/convert/file, /v1/convert/source, async, batch,
        /health, /version."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app

            app = create_app()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                resp = await c.get("/openapi.json")
            schema = resp.json()
            paths = set(schema["paths"].keys())

            required = {
                "/v1/convert/file",
                "/v1/convert/source",
                "/v1/convert/file/async",
                "/v1/convert/source/async",
                "/v1/convert/source/batch",
                "/v1/status/poll/{task_id}",
                "/v1/result/{task_id}",
                "/health",
                "/version",
            }
            missing = required - paths
            assert not missing, f"Missing routes in OpenAPI: {missing}"
        import asyncio
        asyncio.run(run())
    test("Official client: OpenAPI schema declares all 9 required routes", t_official_openapi_routes)

    # ───────────────────────────────────────────────────────────────────
    # 26. Response Content-Type is application/json
    # ───────────────────────────────────────────────────────────────────
    def t_official_response_content_type():
        """The official client expects JSON responses. Our endpoint must
        return Content-Type: application/json."""
        received = []
        resp = _http_post(
            data={"to_formats": ["md"]},
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct, \
            f"Expected application/json, got {ct}"
    test("Official client: response Content-Type is application/json", t_official_response_content_type)

    # ───────────────────────────────────────────────────────────────────
    # 27. Round-trip: encode → send → parse response with official model
    # ───────────────────────────────────────────────────────────────────
    def t_official_full_roundtrip():
        """Full round-trip: encode options with official client → send to
        our endpoint → parse our response with official client's model.
        This is the ultimate compatibility test."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat, ConversionStatus
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        opts = ConvertDocumentsOptions(to_formats=[OutputFormat.MARKDOWN])
        encoded = _official_encode(opts)

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200

        # Parse our response with the official client's model
        parsed = OfficialResp.model_validate(resp.json())
        assert parsed.document.filename == "t.png"
        assert parsed.document.md_content == "# T"
        assert parsed.status == ConversionStatus.SUCCESS
    test("Official client: full round-trip (encode → send → parse response)", t_official_full_roundtrip)

    # ───────────────────────────────────────────────────────────────────
    # 28. Health-check alias endpoint
    # ───────────────────────────────────────────────────────────────────
    def t_official_health_check_alias():
        """The /health-check endpoint is an alias for /health. Some
        platforms use this path. Official client uses /health, but
        /health-check should also work."""
        from docling.datamodel.service.responses import HealthCheckResponse

        resp = _http_get("/health-check")
        assert resp.status_code == 200
        parsed = HealthCheckResponse.model_validate_json(resp.text)
        assert parsed.status == "ok"
    test("Official client: /health-check alias returns parseable response", t_official_health_check_alias)

    # ───────────────────────────────────────────────────────────────────
    # 29. CORS headers present (official client may run in browser)
    # ───────────────────────────────────────────────────────────────────
    def t_official_cors_headers():
        """The app has CORSMiddleware configured. The official client may
        be used from a browser context, so CORS headers must be present."""
        async def run():
            from httpx import ASGITransport, AsyncClient
            from api_compat.docling_api.app import create_app

            app = create_app()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://t",
            ) as c:
                # CORS preflight
                resp = await c.options(
                    "/v1/convert/file",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
            assert resp.status_code == 200, f"CORS preflight: {resp.status_code}"
            assert "access-control-allow-origin" in {k.lower() for k in resp.headers}
        import asyncio
        asyncio.run(run())
    test("Official client: CORS preflight returns allow-origin header", t_official_cors_headers)

    # ───────────────────────────────────────────────────────────────────
    # 30. _serialize_convert_options full pipeline
    # ───────────────────────────────────────────────────────────────────
    def t_official_serialize_pipeline():
        """The official client's _serialize_convert_options() calls
        model_dump(mode='json', exclude_none=True). Verify this produces
        the expected dict for a variety of options."""
        from docling.service_client.client import _BaseDoclingServiceClient
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat

        # _serialize_convert_options is a static method
        opts = ConvertDocumentsOptions(
            to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
            do_ocr=False,
            force_ocr=True,
        )

        # Call it the way the client does
        serialized = opts.model_dump(mode="json", exclude_none=True)
        assert serialized["to_formats"] == ["md", "json"]
        assert serialized["do_ocr"] is False
        assert serialized["force_ocr"] is True

        # Then _form_encode_options
        encoded = _BaseDoclingServiceClient._form_encode_options(serialized)
        assert encoded["to_formats"] == ["md", "json"]
        assert encoded["do_ocr"] is False
        assert encoded["force_ocr"] is True
    test("Official client: _serialize → _form_encode pipeline produces correct types", t_official_serialize_pipeline)

    # ───────────────────────────────────────────────────────────────────
    # 31. Exclude_none behavior — None fields not sent
    # ───────────────────────────────────────────────────────────────────
    def t_official_exclude_none():
        """model_dump(exclude_none=True) must NOT include fields that are
        None. The official client uses this to avoid sending null fields."""
        from docling.datamodel.service.options import ConvertDocumentsOptions

        opts = ConvertDocumentsOptions()  # all defaults
        serialized = opts.model_dump(mode="json", exclude_none=True)

        # ocr_custom_config defaults to None → should NOT be in serialized
        assert "ocr_custom_config" not in serialized, \
            "ocr_custom_config (None) should be excluded"
        # document_timeout defaults to None → should NOT be in serialized
        assert "document_timeout" not in serialized, \
            "document_timeout (None) should be excluded"
    test("Official client: exclude_none=True omits None fields from payload", t_official_exclude_none)

    # ───────────────────────────────────────────────────────────────────
    # 32. Multipart boundary and structure correctness
    # ───────────────────────────────────────────────────────────────────
    def t_official_multipart_structure():
        """The multipart body must have proper structure: boundary,
        Content-Disposition headers, and blank line separators."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        import httpx

        opts = ConvertDocumentsOptions()
        encoded = _official_encode(opts)

        req = httpx.Request(
            "POST", "http://test/v1/convert/file",
            files={"file": ("test.png", b"fake", "image/png")},
            data=encoded,
        )
        raw_body = req.read().decode("utf-8", errors="replace")

        # Must have Content-Type header with boundary
        ct = req.headers.get("content-type", "")
        assert "multipart/form-data" in ct, f"Expected multipart, got {ct}"
        assert "boundary=" in ct, "Missing boundary in Content-Type"

        # Body must contain Content-Disposition headers
        assert 'Content-Disposition: form-data; name="file"' in raw_body, \
            "Missing file field Content-Disposition"
        assert 'name="to_formats"' in raw_body, \
            "Missing to_formats field Content-Disposition"

        # File field must have filename
        assert 'filename="test.png"' in raw_body, \
            "Missing filename in file field"
    test("Official client: multipart body has correct structure (boundary, headers, filename)", t_official_multipart_structure)

    # ───────────────────────────────────────────────────────────────────
    # 33. Response with timings is parseable
    # ───────────────────────────────────────────────────────────────────
    def t_official_response_with_timings():
        """Response with ProfilingItem timings must be parseable."""
        from api_compat.docling_api.schema import (
            ConvertDocumentResponse as OurResp, ExportDocumentResponse,
            ConversionStatus, ProfilingItem, ProfilingScope,
        )
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        our_resp = OurResp(
            document=ExportDocumentResponse(filename="t.png", md_content="# T"),
            status=ConversionStatus.SUCCESS,
            processing_time=0.15,
            timings={
                "layout": ProfilingItem(
                    scope=ProfilingScope.DOCUMENT, count=1, times=[0.05],
                ),
                "ocr": ProfilingItem(
                    scope=ProfilingScope.DOCUMENT, count=1, times=[0.10],
                ),
            },
        )
        json_str = our_resp.model_dump_json()
        parsed = OfficialResp.model_validate_json(json_str)
        assert "layout" in parsed.timings
        assert parsed.timings["layout"].times == [0.05]
    test("Official client: response with ProfilingItem timings parseable", t_official_response_with_timings)

    # ───────────────────────────────────────────────────────────────────
    # 34. Response with all content fields populated
    # ───────────────────────────────────────────────────────────────────
    def t_official_response_all_content():
        """Response with md, json, html, text, doctags content fields
        must be parseable by the official client."""
        from docling_core.types.doc import DoclingDocument
        from api_compat.docling_api.schema import (
            ConvertDocumentResponse as OurResp, ExportDocumentResponse,
            ConversionStatus,
        )
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        doc = DoclingDocument(name="test")
        our_resp = OurResp(
            document=ExportDocumentResponse(
                filename="t.png",
                md_content="# Title\n\nBody text",
                json_content=doc,
                html_content="<html><body>Test</body></html>",
                text_content="Title\nBody text",
                doctags_content="<doctags>",
            ),
            status=ConversionStatus.SUCCESS,
            processing_time=0.2,
        )
        json_str = our_resp.model_dump_json()
        parsed = OfficialResp.model_validate_json(json_str)
        assert parsed.document.md_content == "# Title\n\nBody text"
        assert parsed.document.html_content == "<html><body>Test</body></html>"
        assert parsed.document.text_content == "Title\nBody text"
        assert parsed.document.doctags_content == "<doctags>"
        assert parsed.document.json_content is not None
    test("Official client: response with all 5 content fields parseable", t_official_response_all_content)

    # ───────────────────────────────────────────────────────────────────
    # 35. JSON output format through official encoding
    # ───────────────────────────────────────────────────────────────────
    def t_official_json_format_roundtrip():
        """Request JSON output format via official encoding, verify our
        endpoint receives [JSON] and the response is parseable."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat
        from docling.datamodel.service.responses import (
            ConvertDocumentResponse as OfficialResp,
        )

        opts = ConvertDocumentsOptions(to_formats=[OutputFormat.JSON])
        encoded = _official_encode(opts)

        received = []
        resp = _http_post(
            data=encoded,
            files={"file": ("t.png", b"fake", "image/png")},
            spy_fn=_make_spy(received),
        )
        assert resp.status_code == 200
        assert received[0] == [OutputFormat.JSON]

        # Response should be parseable
        parsed = OfficialResp.model_validate(resp.json())
        assert parsed.document.filename == "t.png"
    test("Official client: JSON output format request → received and response parseable", t_official_json_format_roundtrip)

    # ───────────────────────────────────────────────────────────────────
    # 36. Multiple requests don't interfere (state isolation)
    # ───────────────────────────────────────────────────────────────────
    def t_official_multiple_requests():
        """Send multiple requests through official encoding. Each should
        receive the correct to_formats (no state leakage)."""
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat

        for fmts in ([OutputFormat.MARKDOWN], [OutputFormat.JSON], [OutputFormat.MARKDOWN, OutputFormat.JSON]):
            opts = ConvertDocumentsOptions(to_formats=fmts)
            encoded = _official_encode(opts)

            received = []
            resp = _http_post(
                data=encoded,
                files={"file": ("t.png", b"fake", "image/png")},
                spy_fn=_make_spy(received),
            )
            assert resp.status_code == 200
            assert received[0] == fmts, \
                f"Expected {fmts}, got {received[0]}"
    test("Official client: multiple sequential requests maintain correct state", t_official_multiple_requests)

    # ───────────────────────────────────────────────────────────────────
    # 37. _form_encode_options is deterministic
    # ───────────────────────────────────────────────────────────────────
    def t_official_encoding_deterministic():
        """_form_encode_options should produce identical output for the
        same input options (no random ordering or non-determinism)."""
        from docling.service_client.client import _BaseDoclingServiceClient
        from docling.datamodel.service.options import ConvertDocumentsOptions
        from docling.datamodel.base_models import OutputFormat

        opts = ConvertDocumentsOptions(
            to_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
            do_ocr=False, ocr_lang=["en", "fr"],
        )
        serialized = opts.model_dump(mode="json", exclude_none=True)

        encoded1 = _BaseDoclingServiceClient._form_encode_options(serialized)
        encoded2 = _BaseDoclingServiceClient._form_encode_options(serialized)

        assert encoded1 == encoded2, "Encoding should be deterministic"
    test("Official client: _form_encode_options is deterministic", t_official_encoding_deterministic)

    # ───────────────────────────────────────────────────────────────────
    # 38. Health endpoint responds even without model loaded
    # ───────────────────────────────────────────────────────────────────
    def t_official_health_no_model():
        """The /health endpoint should return 200 even without the
        inference model loaded (it's a liveness probe, not readiness)."""
        from docling.datamodel.service.responses import HealthCheckResponse

        resp = _http_get("/health")
        assert resp.status_code == 200
        # Should be fast (liveness, not readiness)
        parsed = HealthCheckResponse.model_validate_json(resp.text)
        assert parsed.status == "ok"
    test("Official client: /health is a liveness probe (always ok)", t_official_health_no_model)

    # ───────────────────────────────────────────────────────────────────
    # 39. /ready and /readyz readiness endpoints
    # ───────────────────────────────────────────────────────────────────
    def t_official_ready_endpoints():
        """The /ready and /readyz endpoints should return 200 (ready) or
        503 (not ready). In test mode without a loaded model, 503 is
        expected. 200 responses have a 'status' field; 503 responses
        have a 'detail' field (from HTTPException)."""
        for path in ["/ready", "/readyz"]:
            resp = _http_get(path)
            assert resp.status_code in (200, 503), \
                f"{path} should return 200 or 503, got {resp.status_code}"
            data = resp.json()
            if resp.status_code == 200:
                assert "status" in data, f"{path} 200 response missing 'status': {data}"
            else:
                assert "detail" in data, f"{path} 503 response missing 'detail': {data}"
    test("Official client: /ready and /readyz return 200/503 with proper response body", t_official_ready_endpoints)

    # ───────────────────────────────────────────────────────────────────
    # 40. _livez endpoint
    # ───────────────────────────────────────────────────────────────────
    def t_official_livez():
        """The /livez endpoint should return 200 with status='ok'."""
        from docling.datamodel.service.responses import HealthCheckResponse

        resp = _http_get("/livez")
        assert resp.status_code == 200
        parsed = HealthCheckResponse.model_validate_json(resp.text)
        assert parsed.status == "ok"
    test("Official client: /livez returns parseable HealthCheckResponse", t_official_livez)

    # ─── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
