# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Repo is **pre-code**. Only `PROJETO.md` exists — read it in full before any work. It contains the motivation, architectural decisions, MVP scope, open questions, and the suggested implementation order. Several decisions are explicitly blocked on the 8 open questions in `PROJETO.md` — don't start scaffolding code (Dockerfile, worker, loader) until those are answered.

## What this project is

Isolated image-upscaling service. Sister project to `~/projetos/inemaimg/` (realtime generation). Split out because the batch workload (~11M photos) would lock the shared GPU for months and starve `inemaimg`'s realtime path used by `timesmkt3`.

Two surfaces planned:
- **Batch worker** (primary): processes the 11M-photo backlog with checkpoint/resume, idempotency, VRAM-bounded parallelism.
- **HTTP server** (secondary): `POST /upscale` for ad-hoc/agent integration.

## Architectural constraints (do not relitigate without cause)

- **Stack copied, not imported, from `inemaimg`**: same Docker base (aarch64 / CUDA 13 / Blackwell sm_120, NGC PyTorch arm64-sbsa), same FastAPI + loader pattern. Copy the pattern from `~/projetos/inemaimg/`; do not add it as a dependency.
- **Spandrel** is the SR engine — loads arbitrary `.pth` weights (Real-ESRGAN, ESRGAN, SwinIR, HAT, DAT, …). Weights come from **OpenModelDB**, stored in `models/upscale/`.
- **Upscayl is rejected** as a runtime dep (Electron + Vulkan x86_64, incompatible with aarch64/CUDA). Reference only.
- **MVP = 1 model**: `4x-UltraSharp`. No `model` field in the API until a 2nd model exists. No `tile_size` exposed — compute from free VRAM.
- **Input cap**: ≤4096px on the long side, return 413 otherwise. Output is PNG (avoid recompressing JPEG artifacts).
- **GPU is shared** with `inemaimg` via the nvidia runtime, no MPS. Batch must support pause-on-demand or be windowed to off-hours; assume contention by default.

## References

- `PROJETO.md` — source of truth for scope, decisions, and open questions.
- `~/projetos/inemaimg/` — pattern source for Dockerfile, loaders, compose layout.
- `~/projetos/inemaimg/UPSCAYL.md` — prior analysis that motivated the Upscayl rejection.
- Spandrel: https://github.com/chaiNNer-org/spandrel
- OpenModelDB: https://openmodeldb.info
