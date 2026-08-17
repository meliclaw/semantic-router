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

## Fuera de este repo (Capas 2–3)

Waterfall ReMe→SurrealDB→Graphiti→Fuseki y fusión RRF **no** viven aquí. Código nuevo y separable de Meliclaw, Apache 2.0, en `meliclaw-knowledge-mcp`.

## No portado (v1)

Pinecone, aurelio-sdk, LiteLLM, CLIP/ViT, dynamic routes / function calling, notebooks LangChain.
