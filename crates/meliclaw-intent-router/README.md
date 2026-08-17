# Meliclaw Intent Router

Crate: **`meliclaw-intent-router`**. Rama: **`meliclaw-intent-router-main`**.

Intent classifier (Capa 1) de la plataforma Meliclaw. Port Rust del algoritmo de
[aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)
(MIT, Copyright 2024 Aurelio AI). Atribución factual en `THIRD_PARTY_NOTICES.md`.

No es el Memory Router completo. Waterfall ReMe→SurrealDB→Graphiti→Fuseki y RRF
(Capas 2–3) van en `meliclaw-knowledge-mcp` (código propio, Apache 2.0).

Marca comercial: **Meliclaw Intent Classifier**. No usar el nombre del proyecto
upstream ni de su autor en material de venta.

## Crate

```rust
use meliclaw_intent_router::{
    memory_intent_routes, HashDenseEncoder, OnnxEncoder, SemanticRouter, SyncMode,
};

# async fn demo() -> meliclaw_intent_router::Result<()> {
let router = SemanticRouter::builder()
    .encoder(OnnxEncoder::from_model("nomic-embed-text-v1.5")?)
    .routes(memory_intent_routes())
    .auto_sync(SyncMode::Local)
    .build()
    .await?;
let choice = router.route("¿Cuál es el NIF del cliente Y?").await?;
# let _ = choice;
# Ok(())
# }
```

Sin modelo ONNX, `HashDenseEncoder` cubre tests y bootstrap local.

## Servicio HTTP

Crate `meliclaw-intent-router-service`:

- `GET /health`
- `POST /v1/route` `{"query":"..."}`
- `GET|PUT /v1/routes`

Bind: `MELICLAW_INTENT_BIND` (default `0.0.0.0:8091`).

## Features

| Feature | Qué habilita |
|---|---|
| `local-index` (default) | `LocalIndex` |
| `hybrid` (default) | BM25 / TF-IDF / `HybridRouter` |
| `onnx-embed` | carga de `.onnx` (ORT nativo, por despliegue) |
| `openai` / `ollama` | embeddings HTTP |
| `postgres` | pgvector — imagen `pgvector/pgvector:pg17` |
| `qdrant` | Qdrant REST on-prem |

Pinecone, aurelio-sdk y LiteLLM no van en v1.

## Compliance

`main` es espejo del upstream. Trabajo solo en `meliclaw-intent-router-main`.
`LICENSE` MIT de Aurelio AI se conserva. Ver `MELICLAW_PATCHES.md`.
