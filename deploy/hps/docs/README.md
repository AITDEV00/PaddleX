# PaddleX HPS — Documentation Index

> **Location:** `PaddleX/deploy/hps/docs/`
> **Purpose:** Architecture, design, and client-adapter docs for custom HPS work
> (kept separate from upstream `PaddleX/docs/` to avoid merge conflicts).

---

## Start here

| Document | Read when |
|----------|-----------|
| [**API_REFERENCE.md**](API_REFERENCE.md) | You want to talk to the server: every `/v1` path, the `debug` param, response shapes, env vars, and how to build a Docling client adapter. **Self-contained — readable from any directory.** |
| [**architecture/backend_architecture.md**](architecture/backend_architecture.md) | Overview of the pluggable backend refactor (`direct` vs `triton`), `/v1/models`, and the requirements split. |
| [**architecture/adding_a_backend.md**](architecture/adding_a_backend.md) | Step-by-step guide to add a **new inference backend** (`InferenceBackend` subclass) without touching HTTP code. |

## Directory layout

```
docs/
├── API_REFERENCE.md              ← the client/adapter guide (start here)
├── README.md                     ← this index
├── architecture/                 ← stable architecture docs
│   ├── backend_architecture.md
│   └── adding_a_backend.md
├── design/                       ← design plans & decisions
│   ├── api_compat_methodology.md
│   ├── lean_image_design_plan.md
│   └── pp_doclayoutv3_fp8_model_weights.md
└── scratchpads/                  ← dev/analysis notes (best-effort, not canonical)
    ├── architecture_latency_map.md
    ├── code_smell_detection_technique.md
    ├── gpu_util_logic_map.md
    ├── latency_analysis_scratchpad.md
    ├── latency_by_layer.md
    ├── logic_mapping_technique.md
    └── rtx5090_fp8_optimization_scratchpad.md
```

## When to add docs here

- Architecture or design docs for code under `deploy/hps/`.
- Client/API adapter guides (see `API_REFERENCE.md`).
- Methodology or "how to build X" guides (e.g., adding an API compat layer).
- Custom deployment / benchmark notes not covered by upstream `PaddleX/docs/`.

**Do NOT** put upstream PaddlePaddle documentation here — that goes in
`PaddleX/docs/` to stay in sync with `release/3.7`.

## Document conventions

- Every doc is **self-contained**: file references use absolute workspace paths
  so a doc can be read from any directory.
- Prefer linking to `../` relative files within `deploy/hps/docs/` for in-site
  navigation, and absolute `/home/jyao/ADEO/OCR/PaddleX/deploy/hps/...` paths for
  source files.
- Keep scratchpads in `scratchpads/`; promote to `architecture/` or `design/`
  only when a design is settled and canonical.
