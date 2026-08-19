# Third-Party Notices

Meliclaw Intent Router incluye el siguiente software de terceros. Cada proyecto
permanece bajo su propia licencia; nada en este archivo altera esos términos.

Este archivo se regenera con `scripts/gen-third-party-notices.sh` (cargo-about
cuando está instalado). La entrada de origen del algoritmo se mantiene a mano
porque el port Rust es obra derivada, no un crate de crates.io.

---

## Origen del algoritmo (obra derivada)

**Proyecto:** semantic-router (biblioteca Python)
**Origen:** https://github.com/aurelio-labs/semantic-router
**Licencia:** MIT License
**Copyright:** Copyright (c) 2024 Aurelio AI
**NOTICE upstream:** ninguno
**Cambios de Meliclaw:** ver MELICLAW_PATCHES.md

Atribución factual de ingeniería. El nombre del proyecto upstream y de su
autor no se usan en material comercial ni como endoso.

**Crate Meliclaw:** `meliclaw-intent-router` (MIT, obra derivada)

---

## Pesos de modelos de embedding (no son el código MIT)

Las licencias de **pesos** son independientes del crate. No se vendorizan en
este repo. Completar la línea de atribución por despliegue.

| Modelo | Id canónico | Dim | Licencia de pesos | Notas |
|---|---|---|---|---|
| nomic-embed-text-v1.5 | `nomic-ai/nomic-embed-text-v1.5` | 768 | ver tarjeta del modelo | Default CPU/edge ONNX |
| bge-m3 | `BAAI/bge-m3` | 1024 | ver tarjeta del modelo | ONNX |
| **Nemotron-3-Embed-1B-BF16** | `nvidia/Nemotron-3-Embed-1B-BF16` | **2048** | **OpenMDW-1.1** (NVIDIA; uso comercial permitido; Linux Foundation). Built with Ministral-3-3B-Instruct-2512 (Apache-2.0). | Capa 1 GPU / HTTP. Tag Ollama/NIM: `nemotron-3-embed-1b`. No hay URL ONNX oficial de NVIDIA. Export comunitario opcional: `kzzalews/Nemotron-3-Embed-1B-BF16-onnx` (`model.onnx`, `fp16/model.onnx`). Verificar checksum contra pesos NVIDIA antes de producción. |
| **Qwen3-Embedding-0.6B** | `Qwen/Qwen3-Embedding-0.6B` | **1024** | **Apache-2.0** (Alibaba/Qwen; uso comercial permitido). | Opción ligera Qwen para Capa 1. No es el default GPU (sigue Nemotron 1B). Tag Ollama: `qwen3-embedding:0.6b` (explícito; el catálogo no mapea `qwen3-embedding` sin tag). No hay URL ONNX oficial de Qwen. Export comunitario: `neuradex/Qwen3-Embedding-0.6B-ONNX` (`model.onnx`; `last_hidden_state` `[B,S,1024]`; last-token + L2, no mean-pool). Verificar checksum contra pesos Qwen. Capa 1 es simétrica: no aplicar `Instruct: …\nQuery:` solo en queries. Sin slices MRL. |
| **Qwen3-Embedding-8B** | `Qwen/Qwen3-Embedding-8B` | **4096** | **Apache-2.0** (Alibaba/Qwen; uso comercial permitido). | Opcional/pesado. No es default. Tag Ollama: `qwen3-embedding:8b`. En la librería oficial Ollama, `qwen3-embedding:latest` es 8B; este catálogo no mapea el tag sin versión (mirrors comunitarios pueden servir 0.6B). ONNX comunitario: `majentik/Qwen3-Embedding-8B-ONNX-FP16` (`model.onnx` + `model.onnx_data`; `last_hidden_state` `[B,S,4096]`; last-token + L2). Misma simetría que 0.6B. |
| **jina-embeddings-v3** | `jinaai/jina-embeddings-v3` | **1024** | **CC-BY-NC-4.0** (Jina; no comercial en la tarjeta HF). | Ligero (~572M). 8192 es max sequence length, no la dim. MRL no nativo rechazado. Pooling mean. Task LoRA Capa 1: classification o ninguna, no retrieval asimétrico. ONNX oficial: `onnx/model.onnx` en el repo Jina. Q4_K_M GGUF: `second-state/jina-embeddings-v3-GGUF` (`jina-embeddings-v3-Q4_K_M.gguf`). Ollama: `hf.co/second-state/jina-embeddings-v3-GGUF:Q4_K_M` (no hay tag en ollama.com/library). |
| **NV-Embed-v2** | `nvidia/NV-Embed-v2` | **4096** | **CC-BY-NC-4.0** (NVIDIA; no comercial). | Opcional/pesado (~7.85B, Mistral-7B). No es default GPU. Pooling latent-attention (capa aprendida; no mean/last-token sobre `last_hidden_state`). No hay GGUF ni tag Ollama (llama.cpp no soporta `NVEmbedModel`). NVIDIA recomienda NeMo Retriever NIM para uso comercial. |
| **Qwen3-VL-Embedding-2B** | `Qwen/Qwen3-VL-Embedding-2B` | **2048** | **Apache-2.0** (Alibaba/Qwen; uso comercial permitido). | Dense single-vector (EOS). Capa 1 **solo texto** (imágenes → Capa 2). Tag Ollama comunitario Q4_K_M: `RizwanMalik/qwen3-vl-embedding-2b:q4_k_m-q8_0`. GGUF: `Rizwan313/Qwen3-VL-Embedding-2B-GGUF` (`qwen3-vl-embedding-2b-Q4_K_M.gguf`). Sin tag oficial ollama.com. Sin prefijo Instruct asimétrico. |
| **Qwen3-VL-Embedding-8B** | `Qwen/Qwen3-VL-Embedding-8B` | **4096** | **Apache-2.0** (Alibaba/Qwen; uso comercial permitido). | Opcional/pesado. Dense EOS, solo texto. GGUF Q4_K_M: `lainsoykaf/Qwen3-VL-Embedding-8B-GGUF` (`Qwen3-VL-Embedding-8B-Q4_K_M.gguf`). El catálogo no mapea `:latest` comunitario. |
| **gte-Qwen2-7B-instruct** | `Alibaba-NLP/gte-Qwen2-7B-instruct` | **3584** | **Apache-2.0** (Alibaba-NLP; uso comercial permitido). | Opcional/pesado. No es default GPU (sigue Nemotron 1B). Dim **3584**, no 4096. Pooling last-token + L2. Sin ONNX oficial. Q4_K_M: `second-state/gte-Qwen2-7B-instruct-GGUF` (`gte-Qwen2-7B-instruct-Q4_K_M.gguf`). Ollama: `hf.co/second-state/gte-Qwen2-7B-instruct-GGUF:Q4_K_M` (también `since2006/gte-Qwen2-7B-instruct:Q4_K_M`). El catálogo no mapea `:latest`. Capa 1 es simétrica: no aplicar `Instruct: …\nQuery:` solo en queries. |

**ColNomic Embed Multimodal 7B** no se cablea en este crate (retrieval visual /
late-interaction, no un `Vec<f32>` para cosine de Capa 1). **Qwen3-Embedding-4B**
tampoco. **Qwen2.5-VL-Instruct** es un VLM de chat, no un encoder. **Qwen3-VL-Reranker-2B/8B**
son cross-encoders (Capas 2–3 / knowledge-mcp). **gte-Qwen1.5-7B-instruct** (4096-d)
y **gte-Qwen2-1.5B-instruct** tampoco.

Texto OpenMDW-1.1: tarjeta del modelo en
https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16
y https://www.linuxfoundation.org/legal/openmdw-license

Texto Apache-2.0 (Qwen3-Embedding / Qwen3-VL-Embedding / gte-Qwen2-7B-instruct): tarjetas
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B ,
https://huggingface.co/Qwen/Qwen3-Embedding-8B ,
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B ,
https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B y
https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct

Texto CC-BY-NC-4.0 (jina-embeddings-v3): https://huggingface.co/jinaai/jina-embeddings-v3
Texto CC-BY-NC-4.0 (NV-Embed-v2): https://huggingface.co/nvidia/NV-Embed-v2

---

## Crates de Rust (runtime)

Regenerar la tabla con:

```bash
scripts/gen-third-party-notices.sh
```

Hasta que cargo-about esté instalado en CI, el deny-list de `deny.toml` es la
barrera bloqueante (AGPL, SSPL, BUSL/BSL, Elastic, LGPL en artefacto).
