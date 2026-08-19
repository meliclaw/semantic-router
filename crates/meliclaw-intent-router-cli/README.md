# `meliclaw-intent-embed`

LocalAI embeddings probe for Meliclaw Capa 1. The binary talks to LocalAI over
the OpenAI-compatible `POST /v1/embeddings` path via `OpenAiEncoder`. It does
**not** use `OllamaEncoder`.

Capa 1 (`SemanticRouter`) still has no image API. Multimodal mode is a
LocalAI-facing probe only.

**Not in this CLI:** `jinaai/jina-embeddings-v3` and `nvidia/NV-Embed-v2`
(CC-BY-NC-4.0). They remain in the library catalog.

LocalAI must be running (default `http://127.0.0.1:8080/v1`).

## Build / run

```bash
cargo run -p meliclaw-intent-router-cli -- --mode text --model bge-m3 --text "hola"
# binary name: meliclaw-intent-embed
```

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--mode text\|multimodal` | required | |
| `--model <id>` | required | Catalog id or alias |
| `--text <string>` | | Combine with `--file` |
| `--file <path>` | | UTF-8 text file |
| stdin | | Used when `--text`/`--file` omitted and stdin is not a TTY |
| `--image <path>` | repeatable | **Multimodal only** |
| `--base-url` | `http://127.0.0.1:8080/v1` | Env `LOCALAI_BASE_URL` |
| `--api-key` | `sk-local` | Env `OPENAI_API_KEY` then `LOCALAI_API_KEY` |
| `--output json\|table` | `json` | |
| `--full-vector` | off | Otherwise preview is the first 8 floats |

Text mode needs text. Multimodal needs text and/or at least one image
(text-only VL is allowed).

## Allowlist

| Model | dim | text | multimodal |
|---|---|---|---|
| `nomic-embed-text-v1.5` | 768 | yes | no |
| `bge-m3` | 1024 | yes | no |
| `bge-small-en-v1.5` | 384 | yes | no |
| `nvidia/Nemotron-3-Embed-1B-BF16` | 2048 | yes | no |
| `Qwen/Qwen3-Embedding-8B` | 4096 | yes | no |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` | 3584 | yes | no |
| `Qwen/Qwen3-VL-Embedding-2B` | 2048 | yes (text path) | yes |
| `Qwen/Qwen3-VL-Embedding-8B` | 4096 | no | yes |

`Qwen3-Embedding-8B` is the text-only 8B encoder, **not** VL-8B.

## Catalog id → LocalAI YAML `name:`

`--model` accepts catalog ids and library aliases (`resolve_dense_model`).
The request `model` field is the short YAML `name:`:

| `--model` (catalog / alias) | Sent to LocalAI (`name:`) |
|---|---|
| `nomic-embed-text-v1.5`, `nomic-ai/nomic-embed-text-v1.5` | `nomic-embed-text-v1.5` |
| `bge-m3`, `BAAI/bge-m3` | `bge-m3` |
| `bge-small-en-v1.5`, `BAAI/bge-small-en-v1.5` | `bge-small-en-v1.5` |
| `nvidia/Nemotron-3-Embed-1B-BF16`, `nemotron-3-embed-1b` | `nemotron-3-embed-1b` |
| `Qwen/Qwen3-Embedding-8B`, `qwen3-embedding-8b`, `qwen3-embedding:8b` | `qwen3-embedding-8b` |
| `Alibaba-NLP/gte-Qwen2-7B-instruct`, `gte-qwen2-7b-instruct` | `gte-qwen2-7b-instruct` |
| `Qwen/Qwen3-VL-Embedding-2B`, `qwen3-vl-embedding-2b` | `qwen3-vl-embedding-2b` |
| `Qwen/Qwen3-VL-Embedding-8B`, `qwen3-vl-embedding-8b` | `qwen3-vl-embedding-8b` |

Match LocalAI model YAML `name:` to the short alias (or add aliases in LocalAI).

Example YAML:

```yaml
name: nomic-embed-text-v1.5
backend: sentencetransformers
embeddings: true
parameters:
  model: nomic-ai/nomic-embed-text-v1.5
```

```yaml
name: bge-m3
backend: sentencetransformers
embeddings: true
parameters:
  model: BAAI/bge-m3
```

```yaml
name: qwen3-embedding-8b
backend: llama-cpp
embeddings: true
context_size: 32768
parameters:
  model: Qwen3-Embedding-8B-Q4_K_M.gguf
```

```yaml
name: gte-qwen2-7b-instruct
backend: llama-cpp
embeddings: true
context_size: 32768
parameters:
  model: gte-Qwen2-7B-instruct-Q4_K_M.gguf
```

```yaml
name: qwen3-vl-embedding-2b
embeddings: true
# Backend depends on the LocalAI build (transformers / llama-cpp / custom).
parameters:
  model: Qwen/Qwen3-VL-Embedding-2B
```

Gallery names such as `qwen3-embedding-8b` already match this table.

## Payloads

**Text** (`DenseEncoder::encode`) — unchanged OpenAI body:

```json
{ "model": "bge-m3", "input": ["hola"] }
```

**Multimodal** (`OpenAiEncoder::encode_with_images`) — one object so LocalAI
returns a single VL vector (not one vector per string). Native LocalAI
`/v1/embeddings` is typically text-only; a VL backend must accept:

```json
{
  "model": "qwen3-vl-embedding-2b",
  "input": {
    "text": "optional caption",
    "images": ["data:image/png;base64,..."]
  }
}
```

Images are data URLs (`data:image/<type>;base64,...`), not LocalAI server
filesystem paths. Image-only requests omit `input.text`.

## Examples

```bash
# Text
cargo run -p meliclaw-intent-router-cli -- \
  --mode text --model nomic-embed-text-v1.5 --text "¿Cuál es el NIF?"

cargo run -p meliclaw-intent-router-cli -- \
  --mode text --model nvidia/Nemotron-3-Embed-1B-BF16 --file utterance.txt --output table

echo "hello" | cargo run -p meliclaw-intent-router-cli -- \
  --mode text --model Qwen/Qwen3-Embedding-8B

# Multimodal (text only still allowed)
cargo run -p meliclaw-intent-router-cli -- \
  --mode multimodal --model Qwen/Qwen3-VL-Embedding-2B --text "a red bike"

cargo run -p meliclaw-intent-router-cli -- \
  --mode multimodal --model qwen3-vl-embedding-8b \
  --text "invoice" --image ./page.png --image ./stamp.jpg
```

JSON output includes `model`, `localai_model`, `dimensions`, `vector_length`,
`l2_norm`, and `preview` (first 8 floats). Pass `--full-vector` to dump the
vector. HTTP errors and dimension mismatches exit non-zero.
