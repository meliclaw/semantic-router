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

## Modelos de embedding locales

| Modelo | Encoder | Dim | Uso |
|---|---|---|---|
| `nomic-embed-text-v1.5` | `OnnxEncoder` | 768 | Default CPU/edge |
| `bge-m3` | `OnnxEncoder` | 1024 | ONNX |
| `nvidia/Nemotron-3-Embed-1B-BF16` | `OnnxEncoder` (GPU) / `OllamaEncoder` (HTTP) | **2048** | **Default GPU Capa 1.** Tag Ollama/NIM: `nemotron-3-embed-1b`. Pesos OpenMDW-1.1. No hay ONNX oficial NVIDIA; `MELICLAW_ONNX_MODEL` apunta a un `.onnx` local (`model.onnx` o `fp16/model.onnx`). CPU casi seguro fuera de 10–30 ms. |
| `Qwen/Qwen3-Embedding-0.6B` | `OnnxEncoder` / `OllamaEncoder` | **1024** | Opción ligera Qwen (hermano Capa 1). Tag Ollama: `qwen3-embedding:0.6b` (explícito). No hay ONNX oficial; export comunitario `neuradex/Qwen3-Embedding-0.6B-ONNX` (`model.onnx`, `last_hidden_state` `[B,S,1024]`, last-token + L2). Sin slices MRL. **No** aplicar prefijo `Instruct: …\nQuery:` solo en queries — utterances Capa 1 son query-like (simétrico). |
| `Qwen/Qwen3-Embedding-8B` | `OnnxEncoder` / `OllamaEncoder` | **4096** | **Opcional / pesado. No es default.** Tag Ollama: `qwen3-embedding:8b`. En ollama.com, `qwen3-embedding:latest` es 8B; el catálogo **no** mapea el tag sin versión (mirrors comunitarios pueden servir 0.6B). ONNX comunitario `majentik/Qwen3-Embedding-8B-ONNX-FP16` (`model.onnx` + `model.onnx_data`, last-token + L2). Misma simetría que 0.6B. |
| `jinaai/jina-embeddings-v3` | `OnnxEncoder` / `OllamaEncoder` | **1024** | Opción ligera (~572M). **8192 es max sequence length, no la dim.** MRL no nativo → `DimensionMismatch`. Pooling mean. Task LoRA Capa 1: **classification** (o ninguna), no `retrieval.query`/`retrieval.passage`. ONNX oficial en el repo Jina (`onnx/model.onnx`). No hay tag en ollama.com/library; Q4_K_M: `hf.co/second-state/jina-embeddings-v3-GGUF:Q4_K_M` (`jina-embeddings-v3-Q4_K_M.gguf`). Pesos CC-BY-NC-4.0. |
| `nvidia/NV-Embed-v2` | `OnnxEncoder` / HTTP | **4096** | **Opcional / pesado (~7.85B). No es default GPU** (sigue Nemotron 1B). Pooling **latent-attention** (no mean/last-token). Pesos **CC-BY-NC-4.0** (no comercial). No hay GGUF/Ollama (llama.cpp no implementa `NVEmbedModel`). |
| `Qwen/Qwen3-VL-Embedding-2B` | `OnnxEncoder` / `OllamaEncoder` | **2048** | Dense single-vector (EOS / last-token). **Solo texto** — Capa 1 no tiene API de imagen. Tag Ollama comunitario Q4_K_M: `RizwanMalik/qwen3-vl-embedding-2b:q4_k_m-q8_0`. GGUF: `Rizwan313/Qwen3-VL-Embedding-2B-GGUF` (`qwen3-vl-embedding-2b-Q4_K_M.gguf`). Sin tag oficial ollama.com. Sin prefijo Instruct asimétrico. Apache-2.0. |
| `Qwen/Qwen3-VL-Embedding-8B` | `OnnxEncoder` / `OllamaEncoder` | **4096** | **Opcional / pesado. No es default.** Dense EOS, solo texto. GGUF Q4_K_M: `lainsoykaf/Qwen3-VL-Embedding-8B-GGUF` (`Qwen3-VL-Embedding-8B-Q4_K_M.gguf`). El catálogo **no** mapea `:latest` comunitario. Apache-2.0. |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` | `OnnxEncoder` / `OllamaEncoder` | **3584** | **Opcional / pesado. No es default GPU** (sigue Nemotron 1B). Dim nativa **3584** (hidden Qwen2-7B), no 4096 (eso es GTE-Qwen1.5-7B). Pooling last-token + L2. Sin URL ONNX oficial. Q4_K_M: `hf.co/second-state/gte-Qwen2-7B-instruct-GGUF:Q4_K_M` (`gte-Qwen2-7B-instruct-Q4_K_M.gguf`); alias ollama.com `since2006/gte-Qwen2-7B-instruct:Q4_K_M`. El catálogo **no** mapea `:latest` (Q78KG untagged es Q8_0). **No** aplicar prefijo `Instruct: …\nQuery:` solo en queries — utterances Capa 1 son query-like (simétrico). Apache-2.0. |

ColNomic Embed Multimodal 7B **no** entra en este crate. Qwen3-Embedding-4B **tampoco**. `Qwen/Qwen2.5-VL-7B-Instruct` es un VLM de chat, no un encoder. `Qwen3-VL-Reranker-2B/8B` son cross-encoders (Capas 2–3 / knowledge-mcp), no Capa 1. `Alibaba-NLP/gte-Qwen1.5-7B-instruct` (4096-d) y `gte-Qwen2-1.5B-instruct` **tampoco**.

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
