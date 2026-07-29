# HPS Documentation Index

> Location: `PaddleX/deploy/hps/docs/`
> Purpose: Architecture and design docs for custom HPS work (kept separate from upstream `PaddleX/docs/` to avoid merge conflicts).

## Documents

| File | Description |
|------|-------------|
| [api_compat_methodology.md](api_compat_methodology.md) | API compatibility layer architecture — function-level logic map, end-to-end request flow, vertical slice build steps, design principles, and concrete plan for the Unstructured IO layer. |

## When to Add Docs Here

- Architecture or design docs for code under `deploy/hps/`
- Methodology or "how to build X" guides (e.g., adding a new API compat layer)
- Custom deployment or benchmark notes not covered by upstream `PaddleX/docs/`

**Do NOT** put upstream PaddlePaddle documentation here — that goes in `PaddleX/docs/` to stay in sync with `release/3.7`.
