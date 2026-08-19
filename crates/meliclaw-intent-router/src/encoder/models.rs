//! Known dense embedding models for the Meliclaw Intent Classifier (Capa 1).
//!
//! Model **weights** are not MIT. NVIDIA Nemotron weights are OpenMDW-1.1;
//! Qwen3-Embedding, Qwen3-VL-Embedding, and gte-Qwen2-7B-instruct weights are
//! Apache-2.0 (Alibaba/Qwen); jina-embeddings-v3 and NV-Embed-v2 weights are
//! CC-BY-NC-4.0. See
//! `THIRD_PARTY_NOTICES.md`. ColNomic Embed Multimodal 7B is not wired
//! here (late-interaction / visual retrieval, not a single dense vector).
//! Qwen2.5-VL-Instruct is a chat VLM, not an encoder. Qwen3-VL-Reranker
//! models are cross-encoders (Capa 2–3), not dense utterance encoders.
//! Modified by Meliclaw, 2026.

use crate::error::{Error, Result};

/// Official Hugging Face / NVIDIA id for Nemotron-3-Embed-1B (BF16).
pub const NEMOTRON_3_EMBED_1B_ID: &str = "nvidia/Nemotron-3-Embed-1B-BF16";
/// NVIDIA NIM short name; also the Ollama / HTTP tag to serve under.
pub const NEMOTRON_3_EMBED_1B_OLLAMA_TAG: &str = "nemotron-3-embed-1b";
/// Community ONNX export (not NVIDIA-hosted). Verify checksum vs NVIDIA weights.
pub const NEMOTRON_3_EMBED_1B_ONNX_EXPORT: &str = "kzzalews/Nemotron-3-Embed-1B-BF16-onnx";
/// Default graph file in the community ONNX export (float32).
pub const NEMOTRON_3_EMBED_1B_ONNX_FILE: &str = "model.onnx";
/// FP16 graph in the community ONNX export (`fp16/model.onnx` + `.data`).
pub const NEMOTRON_3_EMBED_1B_ONNX_FP16_FILE: &str = "fp16/model.onnx";
/// Native embedding width. Matryoshka slices (1024/512) are not accepted here.
pub const NEMOTRON_3_EMBED_1B_DIM: usize = 2048;

/// Official Hugging Face / Qwen id for Qwen3-Embedding-0.6B.
pub const QWEN3_EMBEDDING_0_6B_ID: &str = "Qwen/Qwen3-Embedding-0.6B";
/// Ollama library tag. Prefer the explicit `:0.6b` — untagged `qwen3-embedding`
/// is **not** in this catalog (official Ollama `latest` is 8B; some community
/// mirrors default to 0.6B).
pub const QWEN3_EMBEDDING_0_6B_OLLAMA_TAG: &str = "qwen3-embedding:0.6b";
/// Community ONNX export (not Qwen-hosted). `last_hidden_state` is `[B,S,1024]`.
/// Apply [`EmbeddingPooling::LastTokenL2`]. Verify checksum vs Qwen weights.
pub const QWEN3_EMBEDDING_0_6B_ONNX_EXPORT: &str = "neuradex/Qwen3-Embedding-0.6B-ONNX";
/// Graph file in the 0.6B community ONNX export.
pub const QWEN3_EMBEDDING_0_6B_ONNX_FILE: &str = "model.onnx";
/// Native embedding width. MRL slices (anything other than 1024) are rejected.
pub const QWEN3_EMBEDDING_0_6B_DIM: usize = 1024;

/// Official Hugging Face / Qwen id for Qwen3-Embedding-8B.
pub const QWEN3_EMBEDDING_8B_ID: &str = "Qwen/Qwen3-Embedding-8B";
/// Ollama library tag. Prefer the explicit `:8b`. Official `qwen3-embedding`
/// / `qwen3-embedding:latest` is 8B on ollama.com, but this catalog does not
/// map the untagged name so a community 0.6B pull cannot silently look like 8B.
pub const QWEN3_EMBEDDING_8B_OLLAMA_TAG: &str = "qwen3-embedding:8b";
/// Community ONNX export (not Qwen-hosted). Output is unpooled
/// `last_hidden_state` `[B,S,4096]`; apply [`EmbeddingPooling::LastTokenL2`].
pub const QWEN3_EMBEDDING_8B_ONNX_EXPORT: &str = "majentik/Qwen3-Embedding-8B-ONNX-FP16";
/// Graph file in the 8B community ONNX export (`model.onnx` + `model.onnx_data`).
pub const QWEN3_EMBEDDING_8B_ONNX_FILE: &str = "model.onnx";
/// Native embedding width. MRL slices are not accepted here.
pub const QWEN3_EMBEDDING_8B_DIM: usize = 4096;

/// Official Hugging Face / Jina id. Native width is **1024**, not 8192
/// (8192 is max sequence length). ~572M params.
pub const JINA_EMBEDDINGS_V3_ID: &str = "jinaai/jina-embeddings-v3";
/// Documented Ollama Hugging Face GGUF pull (not an ollama.com library tag —
/// `ollama.com/library/jina-embeddings-v3` does not exist). Explicit `:Q4_K_M`.
pub const JINA_EMBEDDINGS_V3_OLLAMA_TAG: &str = "hf.co/second-state/jina-embeddings-v3-GGUF:Q4_K_M";
/// Community GGUF repo (Q4_K_M artifact). Not Jina-hosted.
pub const JINA_EMBEDDINGS_V3_GGUF_REPO: &str = "second-state/jina-embeddings-v3-GGUF";
/// Q4_K_M file in [`JINA_EMBEDDINGS_V3_GGUF_REPO`].
pub const JINA_EMBEDDINGS_V3_GGUF_Q4_K_M_FILE: &str = "jina-embeddings-v3-Q4_K_M.gguf";
/// Official ONNX graph inside the Jina repo (`onnx/model.onnx` + `.data`).
pub const JINA_EMBEDDINGS_V3_ONNX_FILE: &str = "onnx/model.onnx";
/// Official FP16 ONNX graph (`onnx/model_fp16.onnx`).
pub const JINA_EMBEDDINGS_V3_ONNX_FP16_FILE: &str = "onnx/model_fp16.onnx";
/// Native embedding width. MRL slices (32…768) are rejected.
pub const JINA_EMBEDDINGS_V3_DIM: usize = 1024;

/// Official Hugging Face / NVIDIA id for NV-Embed-v2 (~7.85B, Mistral-7B).
pub const NV_EMBED_V2_ID: &str = "nvidia/NV-Embed-v2";
/// Native embedding width. There is no GGUF/Ollama path (llama.cpp does not
/// implement `NVEmbedModel` latent-attention pooling).
pub const NV_EMBED_V2_DIM: usize = 4096;

/// Official Hugging Face / Qwen id for Qwen3-VL-Embedding-2B.
/// Dense single-vector (EOS / last-token). Text path only — Capa 1 has no image API.
pub const QWEN3_VL_EMBEDDING_2B_ID: &str = "Qwen/Qwen3-VL-Embedding-2B";
/// Community Ollama tag with Q4_K_M weights. Not an official ollama.com library
/// model. Do not map untagged `:latest`.
pub const QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG: &str = "RizwanMalik/qwen3-vl-embedding-2b:q4_k_m-q8_0";
/// Community GGUF repo (Q4_K_M artifact). Not Qwen-hosted.
pub const QWEN3_VL_EMBEDDING_2B_GGUF_REPO: &str = "Rizwan313/Qwen3-VL-Embedding-2B-GGUF";
/// Q4_K_M file in [`QWEN3_VL_EMBEDDING_2B_GGUF_REPO`].
pub const QWEN3_VL_EMBEDDING_2B_GGUF_Q4_K_M_FILE: &str = "qwen3-vl-embedding-2b-Q4_K_M.gguf";
/// Native embedding width. MRL slices are rejected.
pub const QWEN3_VL_EMBEDDING_2B_DIM: usize = 2048;

/// Official Hugging Face / Qwen id for Qwen3-VL-Embedding-8B.
/// Dense single-vector (EOS / last-token). Text path only. Optional / heavy.
pub const QWEN3_VL_EMBEDDING_8B_ID: &str = "Qwen/Qwen3-VL-Embedding-8B";
/// Community GGUF repo (Q4_K_M artifact). Not Qwen-hosted. No official Ollama
/// library tag; community `:latest` mirrors are not mapped.
pub const QWEN3_VL_EMBEDDING_8B_GGUF_REPO: &str = "lainsoykaf/Qwen3-VL-Embedding-8B-GGUF";
/// Q4_K_M file in [`QWEN3_VL_EMBEDDING_8B_GGUF_REPO`].
pub const QWEN3_VL_EMBEDDING_8B_GGUF_Q4_K_M_FILE: &str = "Qwen3-VL-Embedding-8B-Q4_K_M.gguf";
/// Native embedding width. MRL slices are rejected.
pub const QWEN3_VL_EMBEDDING_8B_DIM: usize = 4096;

/// Official Hugging Face / Alibaba-NLP id for gte-Qwen2-7B-instruct.
/// Native width is **3584** (Qwen2-7B hidden), not 4096 (that is GTE-Qwen1.5-7B).
/// Optional / heavy. Default GPU stays Nemotron 1B.
pub const GTE_QWEN2_7B_INSTRUCT_ID: &str = "Alibaba-NLP/gte-Qwen2-7B-instruct";
/// Documented Ollama Hugging Face GGUF pull (not an ollama.com/library model).
/// Explicit `:Q4_K_M`. Do not map untagged `:latest` (community defaults vary;
/// e.g. Q78KG untagged is Q8_0).
pub const GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG: &str =
    "hf.co/second-state/gte-Qwen2-7B-instruct-GGUF:Q4_K_M";
/// Community ollama.com Q4_K_M tag. Explicit `:Q4_K_M` only.
pub const GTE_QWEN2_7B_INSTRUCT_OLLAMA_COMMUNITY_Q4_K_M: &str =
    "since2006/gte-Qwen2-7B-instruct:Q4_K_M";
/// Community GGUF repo (Q4_K_M artifact). Not Alibaba-hosted.
pub const GTE_QWEN2_7B_INSTRUCT_GGUF_REPO: &str = "second-state/gte-Qwen2-7B-instruct-GGUF";
/// Q4_K_M file in [`GTE_QWEN2_7B_INSTRUCT_GGUF_REPO`].
pub const GTE_QWEN2_7B_INSTRUCT_GGUF_Q4_K_M_FILE: &str = "gte-Qwen2-7B-instruct-Q4_K_M.gguf";
/// Native embedding width. 4096 and other MRL/wrong dims are rejected.
pub const GTE_QWEN2_7B_INSTRUCT_DIM: usize = 3584;

const DEFAULT_THRESHOLD: f32 = 0.3;

/// How to reduce a token-wise `last_hidden_state` `[B, S, D]` to one vector.
///
/// Qwen3-Embedding ONNX graphs emit hidden states, not pooled embeddings.
/// nomic / bge / Nemotron / jina-v3 keep **mean** pooling.
/// Qwen3, Qwen3-VL-Embedding, and GTE-Qwen2-7B-instruct use **last non-pad
/// token + L2** (EOS / last token).
/// NV-Embed-v2 uses a learned **latent-attention** layer that cannot be
/// reconstructed from `last_hidden_state` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPooling {
    /// Mean over non-pad tokens, then L2. nomic, bge, Nemotron, jina-v3.
    Mean,
    /// Last non-pad token (left-pad → last position), then L2.
    /// Qwen3-Embedding, Qwen3-VL-Embedding, and GTE-Qwen2-7B-instruct (EOS).
    LastTokenL2,
    /// Learned latent-attention pooling (NV-Embed). The serving graph must
    /// already emit the pooled vector; do not mean-pool or last-token-pool.
    LatentAttention,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseModelSpec {
    pub id: &'static str,
    pub dimensions: usize,
    pub score_threshold: f32,
    pub pooling: EmbeddingPooling,
}

fn spec(id: &'static str, dimensions: usize, pooling: EmbeddingPooling) -> DenseModelSpec {
    DenseModelSpec {
        id,
        dimensions,
        score_threshold: DEFAULT_THRESHOLD,
        pooling,
    }
}

/// Resolve a caller-supplied model name or alias to the catalog entry.
///
/// Unknown names return `None` so encoders keep their historical defaults.
///
/// Capa 1 utterances **are query-like**. Do not apply Qwen / GTE-Qwen2
/// retrieval prefix `Instruct: …\nQuery:` on queries only — encode queries
/// and documents the same way (no asymmetric prompts). For jina-v3 use the
/// **classification** task LoRA (or none), not `retrieval.query` /
/// `retrieval.passage`.
pub fn resolve_dense_model(name: &str) -> Option<DenseModelSpec> {
    let spec = match name {
        "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
            spec("nomic-embed-text-v1.5", 768, EmbeddingPooling::Mean)
        }
        "bge-m3" | "BAAI/bge-m3" => spec("bge-m3", 1024, EmbeddingPooling::Mean),
        "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => {
            spec("bge-small-en-v1.5", 384, EmbeddingPooling::Mean)
        }
        "nvidia/Nemotron-3-Embed-1B-BF16"
        | "Nemotron-3-Embed-1B-BF16"
        | "nemotron-3-embed-1b"
        | "nemotron-3-embed-1b-bf16"
        | "nvidia/nemotron-3-embed-1b"
        | "kzzalews/Nemotron-3-Embed-1B-BF16-onnx" => spec(
            NEMOTRON_3_EMBED_1B_ID,
            NEMOTRON_3_EMBED_1B_DIM,
            EmbeddingPooling::Mean,
        ),
        "Qwen/Qwen3-Embedding-0.6B"
        | "Qwen3-Embedding-0.6B"
        | "qwen3-embedding-0.6b"
        | "qwen3-embedding:0.6b"
        | "qwen/qwen3-embedding-0.6b"
        | "neuradex/Qwen3-Embedding-0.6B-ONNX" => spec(
            QWEN3_EMBEDDING_0_6B_ID,
            QWEN3_EMBEDDING_0_6B_DIM,
            EmbeddingPooling::LastTokenL2,
        ),
        "Qwen/Qwen3-Embedding-8B"
        | "Qwen3-Embedding-8B"
        | "qwen3-embedding-8b"
        | "qwen3-embedding:8b"
        | "qwen/qwen3-embedding-8b"
        | "majentik/Qwen3-Embedding-8B-ONNX-FP16" => spec(
            QWEN3_EMBEDDING_8B_ID,
            QWEN3_EMBEDDING_8B_DIM,
            EmbeddingPooling::LastTokenL2,
        ),
        "jinaai/jina-embeddings-v3"
        | "jina-embeddings-v3"
        | "jina/jina-embeddings-v3"
        | "hf.co/second-state/jina-embeddings-v3-GGUF:Q4_K_M"
        | "second-state/jina-embeddings-v3-GGUF" => spec(
            JINA_EMBEDDINGS_V3_ID,
            JINA_EMBEDDINGS_V3_DIM,
            EmbeddingPooling::Mean,
        ),
        "nvidia/NV-Embed-v2" | "NV-Embed-v2" | "nv-embed-v2" | "nvidia/nv-embed-v2" => spec(
            NV_EMBED_V2_ID,
            NV_EMBED_V2_DIM,
            EmbeddingPooling::LatentAttention,
        ),
        "Qwen/Qwen3-VL-Embedding-2B"
        | "Qwen3-VL-Embedding-2B"
        | "qwen3-vl-embedding-2b"
        | "qwen/qwen3-vl-embedding-2b"
        | "RizwanMalik/qwen3-vl-embedding-2b:q4_k_m-q8_0"
        | "Rizwan313/Qwen3-VL-Embedding-2B-GGUF" => spec(
            QWEN3_VL_EMBEDDING_2B_ID,
            QWEN3_VL_EMBEDDING_2B_DIM,
            EmbeddingPooling::LastTokenL2,
        ),
        "Qwen/Qwen3-VL-Embedding-8B"
        | "Qwen3-VL-Embedding-8B"
        | "qwen3-vl-embedding-8b"
        | "qwen/qwen3-vl-embedding-8b"
        | "lainsoykaf/Qwen3-VL-Embedding-8B-GGUF" => spec(
            QWEN3_VL_EMBEDDING_8B_ID,
            QWEN3_VL_EMBEDDING_8B_DIM,
            EmbeddingPooling::LastTokenL2,
        ),
        "Alibaba-NLP/gte-Qwen2-7B-instruct"
        | "gte-Qwen2-7B-instruct"
        | "gte-qwen2-7b-instruct"
        | "alibaba-nlp/gte-qwen2-7b-instruct"
        | "hf.co/second-state/gte-Qwen2-7B-instruct-GGUF:Q4_K_M"
        | "second-state/gte-Qwen2-7B-instruct-GGUF"
        | "since2006/gte-Qwen2-7B-instruct:Q4_K_M" => spec(
            GTE_QWEN2_7B_INSTRUCT_ID,
            GTE_QWEN2_7B_INSTRUCT_DIM,
            EmbeddingPooling::LastTokenL2,
        ),
        _ => return None,
    };
    Some(spec)
}

/// Error if `name` is a known model and `dim` is not its native width.
pub fn require_dimensions(name: &str, dim: usize) -> Result<()> {
    if let Some(spec) = resolve_dense_model(name) {
        if spec.dimensions != dim {
            return Err(Error::DimensionMismatch {
                model: spec.id.to_string(),
                expected: spec.dimensions,
                got: dim,
            });
        }
    }
    Ok(())
}

/// Error if a produced embedding vector does not match the encoder width.
pub fn expect_embedding_dim(model: &str, expected: usize, vector: &[f32]) -> Result<()> {
    if vector.len() != expected {
        return Err(Error::DimensionMismatch {
            model: model.to_string(),
            expected,
            got: vector.len(),
        });
    }
    Ok(())
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= norm;
    }
}

/// Pool a dense `last_hidden_state` tensor laid out as `[batch, seq, dim]`.
///
/// `attention_mask` is `[batch, seq]` (`0` = pad, non-zero = token).
/// Used when an ONNX graph emits token states instead of a pooled vector.
pub fn pool_last_hidden_state(
    pooling: EmbeddingPooling,
    hidden: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
    attention_mask: &[i64],
) -> Result<Vec<Vec<f32>>> {
    let expected_hidden = batch
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(dim))
        .ok_or_else(|| Error::msg("pooling shape overflow"))?;
    if hidden.len() != expected_hidden {
        return Err(Error::msg(format!(
            "last_hidden_state length {} != batch {batch} * seq {seq} * dim {dim}",
            hidden.len()
        )));
    }
    if attention_mask.len() != batch.saturating_mul(seq) {
        return Err(Error::msg(format!(
            "attention_mask length {} != batch {batch} * seq {seq}",
            attention_mask.len()
        )));
    }
    if batch == 0 || seq == 0 || dim == 0 {
        return Err(Error::msg("pooling requires non-zero batch, seq, and dim"));
    }

    let mut out = Vec::with_capacity(batch);
    match pooling {
        EmbeddingPooling::LatentAttention => {
            return Err(Error::msg(
                "NV-Embed latent-attention pooling is a learned layer; cannot pool last_hidden_state with mean or last-token",
            ));
        }
        EmbeddingPooling::LastTokenL2 => {
            // Official Qwen recipe: if every row is left-padded, take the last
            // position; otherwise take attention_mask.sum() - 1.
            let left_padding = (0..batch).all(|b| attention_mask[b * seq + (seq - 1)] != 0);
            for b in 0..batch {
                let token_idx = if left_padding {
                    seq - 1
                } else {
                    let sum: i64 = attention_mask[b * seq..(b + 1) * seq].iter().copied().sum();
                    usize::try_from(sum.saturating_sub(1).max(0)).unwrap_or(0)
                };
                let start = (b * seq + token_idx.min(seq - 1)) * dim;
                let mut v = hidden[start..start + dim].to_vec();
                l2_normalize(&mut v);
                out.push(v);
            }
        }
        EmbeddingPooling::Mean => {
            for b in 0..batch {
                let mut acc = vec![0.0f32; dim];
                let mut count = 0.0f32;
                for t in 0..seq {
                    if attention_mask[b * seq + t] != 0 {
                        let start = (b * seq + t) * dim;
                        for d in 0..dim {
                            acc[d] += hidden[start + d];
                        }
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    for x in &mut acc {
                        *x /= count;
                    }
                }
                l2_normalize(&mut acc);
                out.push(acc);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemotron_aliases_are_2048() {
        for name in [
            NEMOTRON_3_EMBED_1B_ID,
            "Nemotron-3-Embed-1B-BF16",
            NEMOTRON_3_EMBED_1B_OLLAMA_TAG,
            "nemotron-3-embed-1b-bf16",
            "nvidia/nemotron-3-embed-1b",
            NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, NEMOTRON_3_EMBED_1B_ID, "{name}");
            assert_eq!(spec.dimensions, 2048, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::Mean, "{name}");
        }
    }

    #[test]
    fn nemotron_rejects_mismatched_dim() {
        let err = require_dimensions(NEMOTRON_3_EMBED_1B_ID, 768).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 768);
                assert_eq!(model, NEMOTRON_3_EMBED_1B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn nemotron_accepts_native_dim() {
        require_dimensions(NEMOTRON_3_EMBED_1B_OLLAMA_TAG, 2048).unwrap();
        expect_embedding_dim(NEMOTRON_3_EMBED_1B_ID, 2048, &vec![0.0; 2048]).unwrap();
    }

    #[test]
    fn qwen3_0_6b_aliases_are_1024() {
        for name in [
            QWEN3_EMBEDDING_0_6B_ID,
            "Qwen3-Embedding-0.6B",
            QWEN3_EMBEDDING_0_6B_OLLAMA_TAG,
            "qwen3-embedding-0.6b",
            "qwen/qwen3-embedding-0.6b",
            QWEN3_EMBEDDING_0_6B_ONNX_EXPORT,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, QWEN3_EMBEDDING_0_6B_ID, "{name}");
            assert_eq!(spec.dimensions, 1024, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::LastTokenL2, "{name}");
        }
    }

    #[test]
    fn qwen3_8b_aliases_are_4096() {
        for name in [
            QWEN3_EMBEDDING_8B_ID,
            "Qwen3-Embedding-8B",
            QWEN3_EMBEDDING_8B_OLLAMA_TAG,
            "qwen3-embedding-8b",
            "qwen/qwen3-embedding-8b",
            QWEN3_EMBEDDING_8B_ONNX_EXPORT,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, QWEN3_EMBEDDING_8B_ID, "{name}");
            assert_eq!(spec.dimensions, 4096, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::LastTokenL2, "{name}");
        }
    }

    #[test]
    fn qwen3_rejects_mismatched_dim() {
        let err = require_dimensions(QWEN3_EMBEDDING_0_6B_ID, 768).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 768);
                assert_eq!(model, QWEN3_EMBEDDING_0_6B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        let err = require_dimensions(QWEN3_EMBEDDING_8B_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 4096);
                assert_eq!(got, 1024);
                assert_eq!(model, QWEN3_EMBEDDING_8B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn qwen3_accepts_native_dim() {
        require_dimensions(QWEN3_EMBEDDING_0_6B_OLLAMA_TAG, 1024).unwrap();
        expect_embedding_dim(QWEN3_EMBEDDING_0_6B_ID, 1024, &vec![0.0; 1024]).unwrap();
        require_dimensions(QWEN3_EMBEDDING_8B_OLLAMA_TAG, 4096).unwrap();
        expect_embedding_dim(QWEN3_EMBEDDING_8B_ID, 4096, &vec![0.0; 4096]).unwrap();
    }

    #[test]
    fn qwen3_untagged_and_4b_are_not_in_catalog() {
        // Official Ollama latest is 8B; community mirrors may serve 0.6B.
        // Require explicit :0.6b / :8b so dim cannot silently disagree.
        assert!(resolve_dense_model("qwen3-embedding").is_none());
        assert!(resolve_dense_model("qwen3-embedding:latest").is_none());
        assert!(resolve_dense_model("qwen3-embedding:4b").is_none());
        assert!(resolve_dense_model("Qwen/Qwen3-Embedding-4B").is_none());
    }

    #[test]
    fn unknown_model_allows_any_dim() {
        require_dimensions("custom-local-embed", 512).unwrap();
        assert!(resolve_dense_model("custom-local-embed").is_none());
    }

    #[test]
    fn last_token_l2_differs_from_mean() {
        // [1, 3, 2]: tokens [3,0], [0,4], [5,12]
        let hidden = [3.0f32, 0.0, 0.0, 4.0, 5.0, 12.0];
        let mask = [1i64, 1, 1];
        let last =
            pool_last_hidden_state(EmbeddingPooling::LastTokenL2, &hidden, 1, 3, 2, &mask).unwrap();
        let mean = pool_last_hidden_state(EmbeddingPooling::Mean, &hidden, 1, 3, 2, &mask).unwrap();
        let last_n = (5.0f32 * 5.0 + 12.0 * 12.0).sqrt();
        assert!((last[0][0] - 5.0 / last_n).abs() < 1e-6);
        assert!((last[0][1] - 12.0 / last_n).abs() < 1e-6);
        assert_ne!(last[0], mean[0]);
    }

    #[test]
    fn last_token_respects_right_padding() {
        let hidden = [3.0f32, 0.0, 0.0, 4.0, 9.0, 9.0];
        let mask = [1i64, 1, 0];
        let last =
            pool_last_hidden_state(EmbeddingPooling::LastTokenL2, &hidden, 1, 3, 2, &mask).unwrap();
        assert!((last[0][0] - 0.0).abs() < 1e-6);
        assert!((last[0][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn last_token_left_padding_takes_last_position() {
        let hidden = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mask = [0i64, 0, 1];
        let last =
            pool_last_hidden_state(EmbeddingPooling::LastTokenL2, &hidden, 1, 3, 2, &mask).unwrap();
        assert!((last[0][0] - 0.0).abs() < 1e-6);
        assert!((last[0][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nomic_and_bge_keep_mean_pooling() {
        assert_eq!(
            resolve_dense_model("nomic-embed-text-v1.5")
                .unwrap()
                .pooling,
            EmbeddingPooling::Mean
        );
        assert_eq!(
            resolve_dense_model("bge-m3").unwrap().pooling,
            EmbeddingPooling::Mean
        );
    }

    #[test]
    fn jina_v3_aliases_are_1024_mean() {
        for name in [
            JINA_EMBEDDINGS_V3_ID,
            "jina-embeddings-v3",
            "jina/jina-embeddings-v3",
            JINA_EMBEDDINGS_V3_OLLAMA_TAG,
            JINA_EMBEDDINGS_V3_GGUF_REPO,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, JINA_EMBEDDINGS_V3_ID, "{name}");
            assert_eq!(spec.dimensions, 1024, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::Mean, "{name}");
        }
    }

    #[test]
    fn jina_v3_rejects_8192_seq_len_as_dim() {
        let err = require_dimensions(JINA_EMBEDDINGS_V3_ID, 8192).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 8192);
                assert_eq!(model, JINA_EMBEDDINGS_V3_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        require_dimensions(JINA_EMBEDDINGS_V3_OLLAMA_TAG, 1024).unwrap();
        expect_embedding_dim(JINA_EMBEDDINGS_V3_ID, 1024, &vec![0.0; 1024]).unwrap();
    }

    #[test]
    fn jina_untagged_latest_is_not_in_catalog() {
        assert!(resolve_dense_model("jina-embeddings-v3:latest").is_none());
        assert!(resolve_dense_model("hf.co/second-state/jina-embeddings-v3-GGUF").is_none());
    }

    #[test]
    fn nv_embed_v2_aliases_are_4096_latent() {
        for name in [
            NV_EMBED_V2_ID,
            "NV-Embed-v2",
            "nv-embed-v2",
            "nvidia/nv-embed-v2",
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, NV_EMBED_V2_ID, "{name}");
            assert_eq!(spec.dimensions, 4096, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::LatentAttention, "{name}");
        }
    }

    #[test]
    fn nv_embed_v2_rejects_mismatched_dim() {
        let err = require_dimensions(NV_EMBED_V2_ID, 2048).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 4096);
                assert_eq!(got, 2048);
                assert_eq!(model, NV_EMBED_V2_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        require_dimensions(NV_EMBED_V2_ID, 4096).unwrap();
    }

    #[test]
    fn nv_embed_latent_attention_cannot_pool_hidden() {
        let hidden = [1.0f32, 0.0, 0.0, 1.0];
        let mask = [1i64, 1];
        let err =
            pool_last_hidden_state(EmbeddingPooling::LatentAttention, &hidden, 1, 2, 2, &mask)
                .unwrap_err();
        match err {
            Error::Message(m) => assert!(m.contains("latent-attention"), "{m}"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn qwen3_vl_2b_aliases_are_2048() {
        for name in [
            QWEN3_VL_EMBEDDING_2B_ID,
            "Qwen3-VL-Embedding-2B",
            QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG,
            "qwen3-vl-embedding-2b",
            "qwen/qwen3-vl-embedding-2b",
            QWEN3_VL_EMBEDDING_2B_GGUF_REPO,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, QWEN3_VL_EMBEDDING_2B_ID, "{name}");
            assert_eq!(spec.dimensions, 2048, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::LastTokenL2, "{name}");
        }
    }

    #[test]
    fn qwen3_vl_8b_aliases_are_4096() {
        for name in [
            QWEN3_VL_EMBEDDING_8B_ID,
            "Qwen3-VL-Embedding-8B",
            "qwen3-vl-embedding-8b",
            "qwen/qwen3-vl-embedding-8b",
            QWEN3_VL_EMBEDDING_8B_GGUF_REPO,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, QWEN3_VL_EMBEDDING_8B_ID, "{name}");
            assert_eq!(spec.dimensions, 4096, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::LastTokenL2, "{name}");
        }
    }

    #[test]
    fn qwen3_vl_rejects_mismatched_dim() {
        let err = require_dimensions(QWEN3_VL_EMBEDDING_2B_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 1024);
                assert_eq!(model, QWEN3_VL_EMBEDDING_2B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        let err = require_dimensions(QWEN3_VL_EMBEDDING_8B_ID, 2048).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 4096);
                assert_eq!(got, 2048);
                assert_eq!(model, QWEN3_VL_EMBEDDING_8B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        require_dimensions(QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG, 2048).unwrap();
        expect_embedding_dim(QWEN3_VL_EMBEDDING_2B_ID, 2048, &vec![0.0; 2048]).unwrap();
        require_dimensions(QWEN3_VL_EMBEDDING_8B_ID, 4096).unwrap();
        expect_embedding_dim(QWEN3_VL_EMBEDDING_8B_ID, 4096, &vec![0.0; 4096]).unwrap();
    }

    #[test]
    fn qwen3_vl_untagged_instruct_and_reranker_are_not_in_catalog() {
        assert!(resolve_dense_model("qwen3-vl-embedding").is_none());
        assert!(resolve_dense_model("qwen3-vl-embedding:latest").is_none());
        assert!(resolve_dense_model("RizwanMalik/qwen3-vl-embedding-2b").is_none());
        assert!(resolve_dense_model("RizwanMalik/qwen3-vl-embedding-2b:latest").is_none());
        assert!(resolve_dense_model("batiai/qwen3-vl-embed-2b").is_none());
        assert!(resolve_dense_model("batiai/qwen3-vl-embed-8b:latest").is_none());
        assert!(resolve_dense_model("Qwen/Qwen2.5-VL-7B-Instruct").is_none());
        assert!(resolve_dense_model("Qwen/Qwen3-VL-Reranker-2B").is_none());
        assert!(resolve_dense_model("Qwen/Qwen3-VL-Reranker-8B").is_none());
    }

    #[test]
    fn gte_qwen2_7b_aliases_are_3584() {
        for name in [
            GTE_QWEN2_7B_INSTRUCT_ID,
            "gte-Qwen2-7B-instruct",
            "gte-qwen2-7b-instruct",
            "alibaba-nlp/gte-qwen2-7b-instruct",
            GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG,
            GTE_QWEN2_7B_INSTRUCT_OLLAMA_COMMUNITY_Q4_K_M,
            GTE_QWEN2_7B_INSTRUCT_GGUF_REPO,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, GTE_QWEN2_7B_INSTRUCT_ID, "{name}");
            assert_eq!(spec.dimensions, 3584, "{name}");
            assert_eq!(spec.pooling, EmbeddingPooling::LastTokenL2, "{name}");
        }
    }

    #[test]
    fn gte_qwen2_7b_rejects_4096_and_other_dims() {
        let err = require_dimensions(GTE_QWEN2_7B_INSTRUCT_ID, 4096).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 3584);
                assert_eq!(got, 4096);
                assert_eq!(model, GTE_QWEN2_7B_INSTRUCT_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        let err = require_dimensions(GTE_QWEN2_7B_INSTRUCT_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 3584);
                assert_eq!(got, 1024);
            }
            other => panic!("unexpected error: {other}"),
        }
        require_dimensions(GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG, 3584).unwrap();
        expect_embedding_dim(GTE_QWEN2_7B_INSTRUCT_ID, 3584, &vec![0.0; 3584]).unwrap();
    }

    #[test]
    fn gte_qwen2_untagged_and_qwen15_are_not_in_catalog() {
        assert!(resolve_dense_model("gte-Qwen2-7B-instruct:latest").is_none());
        assert!(resolve_dense_model("gte-qwen2-7b-instruct:latest").is_none());
        assert!(resolve_dense_model("Q78KG/gte-Qwen2-7B-instruct").is_none());
        assert!(resolve_dense_model("Q78KG/gte-Qwen2-7B-instruct:latest").is_none());
        assert!(resolve_dense_model("since2006/gte-Qwen2-7B-instruct").is_none());
        assert!(resolve_dense_model("hf.co/second-state/gte-Qwen2-7B-instruct-GGUF").is_none());
        assert!(resolve_dense_model("Alibaba-NLP/gte-Qwen1.5-7B-instruct").is_none());
        assert!(resolve_dense_model("Alibaba-NLP/gte-Qwen2-1.5B-instruct").is_none());
    }
}
