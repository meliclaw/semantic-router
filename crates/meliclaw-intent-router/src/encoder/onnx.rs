//! ONNX embedding encoder (on-prem default in architecture: nomic / bge-m3).
//!
//! Runtime ONNX session is optional (`onnx-embed`). Without a model file this
//! encoder uses [`HashDenseEncoder`] with the published dimension of the named
//! model so the rest of the stack can boot. Production must set
//! `MELICLAW_ONNX_MODEL` to a real `.onnx` path and enable the feature.
//!
//! NVIDIA Nemotron-3-Embed-1B is a first-class option (2048-d, GPU). There is
//! no NVIDIA-hosted ONNX URL; operators point `MELICLAW_ONNX_MODEL` at a local
//! graph (`model.onnx` or `fp16/model.onnx` from the community export
//! `kzzalews/Nemotron-3-Embed-1B-BF16-onnx`) after checksumming against
//! `nvidia/Nemotron-3-Embed-1B-BF16`.
//!
//! Qwen3-Embedding-0.6B (1024-d) and Qwen3-Embedding-8B (4096-d) are catalog
//! options. There is no official Qwen ONNX URL. Community graphs emit
//! `last_hidden_state` `[B,S,D]` — pool with [`EmbeddingPooling::LastTokenL2`],
//! not mean-pool (nomic / bge / Nemotron / jina-v3 stay mean).
//! jina-embeddings-v3 is 1024-d (8192 is max sequence length). Official ONNX
//! lives in the Jina repo (`onnx/model.onnx`). Use classification LoRA or none,
//! not retrieval-asymmetric. NV-Embed-v2 is 4096-d with latent-attention
//! pooling — do not mean/last-token pool hidden states. Qwen3-VL-Embedding-2B
//! (2048-d) and 8B (4096-d) are dense EOS vectors; Capa 1 is **text only**.
//! gte-Qwen2-7B-instruct is 3584-d (not 4096); last-token + L2. No official
//! ONNX URL. Capa 1 utterances are query-like: do not apply
//! `Instruct: …\nQuery:` only on queries.
//!
//! Model **weights** are not MIT. Complete the license line per deployment.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use super::models::{require_dimensions, resolve_dense_model, EmbeddingPooling};
use super::{DenseEncoder, HashDenseEncoder};
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct OnnxEncoder {
    model_id: String,
    model_path: Option<PathBuf>,
    pooling: EmbeddingPooling,
    inner: HashDenseEncoder,
}

impl OnnxEncoder {
    /// Known Meliclaw embedding models (v5.4 §14.2). Unknown names keep dim 768.
    pub fn from_model(name: &str) -> Result<Self> {
        let (id, dim, threshold, pooling) = match resolve_dense_model(name) {
            Some(spec) => (spec.id, spec.dimensions, spec.score_threshold, spec.pooling),
            None => (name, 768, 0.3, EmbeddingPooling::Mean),
        };
        let path = std::env::var("MELICLAW_ONNX_MODEL")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists());
        Ok(Self {
            model_id: id.to_string(),
            model_path: path,
            pooling,
            inner: HashDenseEncoder::new(id, dim, Some(threshold)),
        })
    }

    /// Same as [`from_model`], but reject a known model when `dim` is wrong.
    ///
    /// Nemotron-3-Embed-1B must be 2048. Qwen3-Embedding-0.6B must be 1024.
    /// Qwen3-Embedding-8B must be 4096. jina-v3 must be 1024 (not 8192).
    /// NV-Embed-v2 must be 4096. Qwen3-VL-Embedding-2B/8B must be 2048/4096.
    /// gte-Qwen2-7B-instruct must be 3584 (not 4096). Unknown names use `dim`
    /// as-is.
    pub fn from_model_with_dimensions(name: &str, dim: usize) -> Result<Self> {
        require_dimensions(name, dim)?;
        let mut enc = Self::from_model(name)?;
        if enc.dimensions() != dim {
            enc.inner = HashDenseEncoder::new(enc.model_id.clone(), dim, Some(0.3));
        }
        Ok(enc)
    }

    pub fn from_path(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::Onnx(format!("model not found: {}", path.display())));
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("onnx");
        // Community export uses `model.onnx`; parent dir or env id still identifies the catalog model.
        let hint = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or(stem);
        require_dimensions(hint, dim)?;
        require_dimensions(stem, dim)?;
        let pooling = resolve_dense_model(hint)
            .or_else(|| resolve_dense_model(stem))
            .map(|s| s.pooling)
            .unwrap_or(EmbeddingPooling::Mean);
        // Full ORT session lands behind feature onnx-embed + native lib.
        // Until then we refuse silent hash fallback when an explicit path is given
        // but the runtime is not compiled in — callers know they asked for ONNX.
        // When ORT is linked, token-state graphs must call
        // `pool_last_hidden_state(self.pooling, …)` so Qwen3 last-token+L2
        // does not silently mean-pool like nomic/bge/Nemotron.
        if cfg!(feature = "onnx-embed") {
            tracing::warn!(
                path = %path.display(),
                "ONNX path provided; native ORT session not linked in this build, using hash stand-in of matching dim"
            );
            Ok(Self {
                model_id: stem.to_string(),
                model_path: Some(path.to_path_buf()),
                pooling,
                inner: HashDenseEncoder::new("onnx", dim, Some(0.3)),
            })
        } else {
            Err(Error::Onnx(
                "rebuild with feature `onnx-embed` and a linked ONNX Runtime to load .onnx files"
                    .into(),
            ))
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    pub fn pooling(&self) -> EmbeddingPooling {
        self.pooling
    }
}

#[async_trait]
impl DenseEncoder for OnnxEncoder {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn encoder_type(&self) -> &str {
        "onnx"
    }
    fn score_threshold(&self) -> Option<f32> {
        self.inner.score_threshold()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.encode(docs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{
        DenseEncoder, GTE_QWEN2_7B_INSTRUCT_DIM, GTE_QWEN2_7B_INSTRUCT_GGUF_REPO,
        GTE_QWEN2_7B_INSTRUCT_ID, GTE_QWEN2_7B_INSTRUCT_OLLAMA_COMMUNITY_Q4_K_M,
        GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG, JINA_EMBEDDINGS_V3_DIM, JINA_EMBEDDINGS_V3_GGUF_REPO,
        JINA_EMBEDDINGS_V3_ID, JINA_EMBEDDINGS_V3_OLLAMA_TAG, NEMOTRON_3_EMBED_1B_DIM,
        NEMOTRON_3_EMBED_1B_ID, NEMOTRON_3_EMBED_1B_OLLAMA_TAG, NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
        NV_EMBED_V2_DIM, NV_EMBED_V2_ID, QWEN3_EMBEDDING_0_6B_DIM, QWEN3_EMBEDDING_0_6B_ID,
        QWEN3_EMBEDDING_0_6B_OLLAMA_TAG, QWEN3_EMBEDDING_0_6B_ONNX_EXPORT, QWEN3_EMBEDDING_8B_DIM,
        QWEN3_EMBEDDING_8B_ID, QWEN3_EMBEDDING_8B_OLLAMA_TAG, QWEN3_EMBEDDING_8B_ONNX_EXPORT,
        QWEN3_VL_EMBEDDING_2B_DIM, QWEN3_VL_EMBEDDING_2B_GGUF_REPO, QWEN3_VL_EMBEDDING_2B_ID,
        QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG, QWEN3_VL_EMBEDDING_8B_DIM,
        QWEN3_VL_EMBEDDING_8B_GGUF_REPO, QWEN3_VL_EMBEDDING_8B_ID,
    };

    #[tokio::test]
    async fn known_model_ids() {
        let e = OnnxEncoder::from_model("nomic-embed-text-v1.5").unwrap();
        assert_eq!(e.model_id(), "nomic-embed-text-v1.5");
        assert_eq!(e.dimensions(), 768);
        assert_eq!(e.pooling(), EmbeddingPooling::Mean);
        let v = e.encode(&[String::from("hola")]).await.unwrap();
        assert_eq!(v[0].len(), 768);
    }

    #[tokio::test]
    async fn nemotron_id_and_dim_2048() {
        for name in [
            NEMOTRON_3_EMBED_1B_ID,
            NEMOTRON_3_EMBED_1B_OLLAMA_TAG,
            NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
            "Nemotron-3-Embed-1B-BF16",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), NEMOTRON_3_EMBED_1B_ID, "{name}");
            assert_eq!(e.dimensions(), NEMOTRON_3_EMBED_1B_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::Mean, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 2048, "{name}");
        }
    }

    #[test]
    fn nemotron_rejects_mismatched_dim() {
        let err =
            OnnxEncoder::from_model_with_dimensions(NEMOTRON_3_EMBED_1B_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 1024);
                assert_eq!(model, NEMOTRON_3_EMBED_1B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn nemotron_accepts_native_dim() {
        let e = OnnxEncoder::from_model_with_dimensions(NEMOTRON_3_EMBED_1B_ID, 2048).unwrap();
        assert_eq!(e.dimensions(), 2048);
    }

    #[tokio::test]
    async fn qwen3_0_6b_id_and_dim_1024() {
        for name in [
            QWEN3_EMBEDDING_0_6B_ID,
            QWEN3_EMBEDDING_0_6B_OLLAMA_TAG,
            QWEN3_EMBEDDING_0_6B_ONNX_EXPORT,
            "Qwen3-Embedding-0.6B",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), QWEN3_EMBEDDING_0_6B_ID, "{name}");
            assert_eq!(e.dimensions(), QWEN3_EMBEDDING_0_6B_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::LastTokenL2, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 1024, "{name}");
        }
    }

    #[tokio::test]
    async fn qwen3_8b_id_and_dim_4096() {
        for name in [
            QWEN3_EMBEDDING_8B_ID,
            QWEN3_EMBEDDING_8B_OLLAMA_TAG,
            QWEN3_EMBEDDING_8B_ONNX_EXPORT,
            "Qwen3-Embedding-8B",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), QWEN3_EMBEDDING_8B_ID, "{name}");
            assert_eq!(e.dimensions(), QWEN3_EMBEDDING_8B_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::LastTokenL2, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 4096, "{name}");
        }
    }

    #[test]
    fn qwen3_rejects_mismatched_dim() {
        let err =
            OnnxEncoder::from_model_with_dimensions(QWEN3_EMBEDDING_0_6B_ID, 768).unwrap_err();
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
        let err = OnnxEncoder::from_model_with_dimensions(QWEN3_EMBEDDING_8B_ID, 1024).unwrap_err();
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
        let e = OnnxEncoder::from_model_with_dimensions(QWEN3_EMBEDDING_0_6B_ID, 1024).unwrap();
        assert_eq!(e.dimensions(), 1024);
        let e = OnnxEncoder::from_model_with_dimensions(QWEN3_EMBEDDING_8B_ID, 4096).unwrap();
        assert_eq!(e.dimensions(), 4096);
    }

    #[tokio::test]
    async fn qwen3_encode_queries_is_symmetric() {
        // Capa 1 utterances are query-like: no Instruct/Query prefix on queries only.
        let e = OnnxEncoder::from_model(QWEN3_EMBEDDING_0_6B_ID).unwrap();
        let docs = vec![String::from("cuál es el NIF")];
        let a = e.encode(&docs).await.unwrap();
        let b = e.encode_queries(&docs).await.unwrap();
        let c = e.encode_documents(&docs).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[tokio::test]
    async fn jina_v3_id_and_dim_1024() {
        for name in [
            JINA_EMBEDDINGS_V3_ID,
            JINA_EMBEDDINGS_V3_OLLAMA_TAG,
            JINA_EMBEDDINGS_V3_GGUF_REPO,
            "jina-embeddings-v3",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), JINA_EMBEDDINGS_V3_ID, "{name}");
            assert_eq!(e.dimensions(), JINA_EMBEDDINGS_V3_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::Mean, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 1024, "{name}");
        }
    }

    #[test]
    fn jina_v3_rejects_8192() {
        let err = OnnxEncoder::from_model_with_dimensions(JINA_EMBEDDINGS_V3_ID, 8192).unwrap_err();
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
        let e = OnnxEncoder::from_model_with_dimensions(JINA_EMBEDDINGS_V3_ID, 1024).unwrap();
        assert_eq!(e.dimensions(), 1024);
    }

    #[tokio::test]
    async fn nv_embed_v2_id_and_dim_4096() {
        let e = OnnxEncoder::from_model(NV_EMBED_V2_ID).unwrap();
        assert_eq!(e.model_id(), NV_EMBED_V2_ID);
        assert_eq!(e.dimensions(), NV_EMBED_V2_DIM);
        assert_eq!(e.pooling(), EmbeddingPooling::LatentAttention);
        let v = e.encode(&[String::from("hola")]).await.unwrap();
        assert_eq!(v[0].len(), 4096);
    }

    #[test]
    fn nv_embed_v2_rejects_mismatched_dim() {
        let err = OnnxEncoder::from_model_with_dimensions(NV_EMBED_V2_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 4096);
                assert_eq!(got, 1024);
                assert_eq!(model, NV_EMBED_V2_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
        let e = OnnxEncoder::from_model_with_dimensions(NV_EMBED_V2_ID, 4096).unwrap();
        assert_eq!(e.dimensions(), 4096);
    }

    #[tokio::test]
    async fn qwen3_vl_2b_id_and_dim_2048() {
        for name in [
            QWEN3_VL_EMBEDDING_2B_ID,
            QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG,
            QWEN3_VL_EMBEDDING_2B_GGUF_REPO,
            "Qwen3-VL-Embedding-2B",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), QWEN3_VL_EMBEDDING_2B_ID, "{name}");
            assert_eq!(e.dimensions(), QWEN3_VL_EMBEDDING_2B_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::LastTokenL2, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 2048, "{name}");
        }
    }

    #[tokio::test]
    async fn qwen3_vl_8b_id_and_dim_4096() {
        for name in [
            QWEN3_VL_EMBEDDING_8B_ID,
            QWEN3_VL_EMBEDDING_8B_GGUF_REPO,
            "Qwen3-VL-Embedding-8B",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), QWEN3_VL_EMBEDDING_8B_ID, "{name}");
            assert_eq!(e.dimensions(), QWEN3_VL_EMBEDDING_8B_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::LastTokenL2, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 4096, "{name}");
        }
    }

    #[test]
    fn qwen3_vl_rejects_mismatched_dim() {
        let err =
            OnnxEncoder::from_model_with_dimensions(QWEN3_VL_EMBEDDING_2B_ID, 1024).unwrap_err();
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
        let err =
            OnnxEncoder::from_model_with_dimensions(QWEN3_VL_EMBEDDING_8B_ID, 2048).unwrap_err();
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
        let e = OnnxEncoder::from_model_with_dimensions(QWEN3_VL_EMBEDDING_2B_ID, 2048).unwrap();
        assert_eq!(e.dimensions(), 2048);
        let e = OnnxEncoder::from_model_with_dimensions(QWEN3_VL_EMBEDDING_8B_ID, 4096).unwrap();
        assert_eq!(e.dimensions(), 4096);
    }

    #[tokio::test]
    async fn qwen3_vl_encode_queries_is_symmetric() {
        let e = OnnxEncoder::from_model(QWEN3_VL_EMBEDDING_2B_ID).unwrap();
        let docs = vec![String::from("cuál es el NIF")];
        let a = e.encode(&docs).await.unwrap();
        let b = e.encode_queries(&docs).await.unwrap();
        let c = e.encode_documents(&docs).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[tokio::test]
    async fn gte_qwen2_7b_id_and_dim_3584() {
        for name in [
            GTE_QWEN2_7B_INSTRUCT_ID,
            GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG,
            GTE_QWEN2_7B_INSTRUCT_OLLAMA_COMMUNITY_Q4_K_M,
            GTE_QWEN2_7B_INSTRUCT_GGUF_REPO,
            "gte-qwen2-7b-instruct",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), GTE_QWEN2_7B_INSTRUCT_ID, "{name}");
            assert_eq!(e.dimensions(), GTE_QWEN2_7B_INSTRUCT_DIM, "{name}");
            assert_eq!(e.pooling(), EmbeddingPooling::LastTokenL2, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 3584, "{name}");
        }
    }

    #[test]
    fn gte_qwen2_7b_rejects_4096() {
        let err =
            OnnxEncoder::from_model_with_dimensions(GTE_QWEN2_7B_INSTRUCT_ID, 4096).unwrap_err();
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
        let e = OnnxEncoder::from_model_with_dimensions(GTE_QWEN2_7B_INSTRUCT_ID, 3584).unwrap();
        assert_eq!(e.dimensions(), 3584);
    }

    #[tokio::test]
    async fn gte_qwen2_encode_queries_is_symmetric() {
        let e = OnnxEncoder::from_model(GTE_QWEN2_7B_INSTRUCT_ID).unwrap();
        let docs = vec![String::from("cuál es el NIF")];
        let a = e.encode(&docs).await.unwrap();
        let b = e.encode_queries(&docs).await.unwrap();
        let c = e.encode_documents(&docs).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}
