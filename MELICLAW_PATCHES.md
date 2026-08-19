# Meliclaw Patches — Intent Router

**Upstream:** https://github.com/aurelio-labs/semantic-router
**Licencia:** MIT (Copyright 2024 Aurelio AI). Sin `NOTICE` en el upstream — nada que propagar.
**Tag de release:** `v0.1.16` (`ec1dec7595eb9688bd2e79abed8ad2d9e11c1da5`)
**Commit base de esta rama:** `a4576168d9589397a7e0c6ff77f5d05469a56e2e` (v0.1.16 + PR #678 dep advisories)
**Último sync:** 2026-08-16
**Rama de trabajo:** `meliclaw-intent-router-main`
**Rama espejo:** `main` (no se toca)
**Crate:** `meliclaw-intent-router`

## Compliance

| Check | Estado |
|---|---|
| Fork privado | **P0 abierto:** `github.com/meliclaw/semantic-router` está público. Convertir a privado. |
| `LICENSE` preservado | Sí — MIT, Copyright 2024 Aurelio AI, sin modificar |
| `NOTICE` upstream | Ninguno |
| Remote `upstream` | `https://github.com/aurelio-labs/semantic-router.git` |
| Anclado a tag | Release `v0.1.16`; rama incluye fix de advisories posterior al tag |

## Cambios

| Fecha | Archivo(s) | Cambio | Motivo |
|---|---|---|---|
| 2026-08-16 | — | Fork inicial + rama `meliclaw-intent-router-main` | Punto de partida |
| 2026-08-16 | `crates/meliclaw-intent-router/**` | Port Rust del Intent Classifier (Capa 1) | Versión propia on-prem |
| 2026-08-16 | `crates/meliclaw-intent-router-service/**` | Servicio Axum `POST /v1/route` | Hermes / knowledge-mcp |
| 2026-08-16 | `scripts/*`, `deny.toml`, `.github/workflows/meliclaw-ci.yml` | CI de licencias e imágenes | Guía Meliclaw §5 |
| 2026-08-16 | `THIRD_PARTY_NOTICES.md` | Atribución factual MIT | Guía §6 |
| 2026-08-17 | `crates/meliclaw-intent-router/src/encoder/{models,onnx,ollama}.rs`, `THIRD_PARTY_NOTICES.md` | Cablear Nemotron-3-Embed-1B (id + dim 2048 + OpenMDW-1.1) en OnnxEncoder / OllamaEncoder | Embedding GPU/HTTP Capa 1 (v5.4 §14.2) |
| 2026-08-19 | `crates/meliclaw-intent-router/src/encoder/{models,onnx,ollama,mod}.rs`, `THIRD_PARTY_NOTICES.md` | Cablear Qwen3-Embedding-0.6B (1024) y 8B (4096); last-token+L2 para ONNX Qwen3; sin 4B ni default 8B | Embedding Qwen Capa 1 (Apache-2.0) |
| 2026-08-19 | `crates/meliclaw-intent-router/src/encoder/{models,onnx,ollama,mod,lib}.rs`, `THIRD_PARTY_NOTICES.md` | Cablear jina-v3 (1024, no 8192), NV-Embed-v2 (4096 latent-attention), Qwen3-VL-Embedding-2B/8B (2048/4096, texto) | Catálogo Capa 1 adicional |
| 2026-08-19 | `crates/meliclaw-intent-router/src/encoder/{models,onnx,ollama,mod,openai}.rs`, `crates/meliclaw-intent-router-cli/src/allowlist.rs`, `THIRD_PARTY_NOTICES.md` | Cablear gte-Qwen2-7B-instruct (3584, no 4096; last-token+L2; Apache-2.0; Q4_K_M hf.co second-state) | Embedding GTE-Qwen2 Capa 1 opcional/pesado |

## Fuera de este repo (Capas 2–3)

Waterfall ReMe→SurrealDB→Graphiti→Fuseki y fusión RRF **no** viven aquí. Código nuevo y separable de Meliclaw, Apache 2.0, en `meliclaw-knowledge-mcp`.

## No portado (v1)

Pinecone, aurelio-sdk, LiteLLM, CLIP/ViT, dynamic routes / function calling, notebooks LangChain.
