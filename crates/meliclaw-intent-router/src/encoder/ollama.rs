//! Ollama / LM Studio / vLLM embeddings. Feature `ollama`.
//!
//! Nemotron-3-Embed-1B is served over this HTTP API (Ollama tag
//! `nemotron-3-embed-1b`, or Hugging Face id `nvidia/Nemotron-3-Embed-1B-BF16`
//! when the proxy forwards that name). Native width is 2048.
//!
//! Qwen3-Embedding uses explicit tags only: `qwen3-embedding:0.6b` (1024-d)
//! and `qwen3-embedding:8b` (4096-d). Untagged `qwen3-embedding` is not in the
//! catalog — official Ollama `latest` is 8B, while some community mirrors
//! default to 0.6B.
//!
//! jina-embeddings-v3 is 1024-d (8192 is max sequence length). There is no
//! ollama.com library tag; pull `hf.co/second-state/jina-embeddings-v3-GGUF:Q4_K_M`.
//! NV-Embed-v2 is 4096-d; there is no GGUF/Ollama path (latent-attention).
//! Qwen3-VL-Embedding-2B (2048-d) community Q4_K_M tag:
//! `RizwanMalik/qwen3-vl-embedding-2b:q4_k_m-q8_0`. 8B has a GGUF Q4_K_M
//! artifact but no official Ollama tag. gte-Qwen2-7B-instruct is 3584-d;
//! Q4_K_M pull `hf.co/second-state/gte-Qwen2-7B-instruct-GGUF:Q4_K_M` (also
//! `since2006/gte-Qwen2-7B-instruct:Q4_K_M`). Untagged `:latest` is not mapped.
//! Capa 1 is text-only and symmetric: do not send Qwen / GTE
//! `Instruct: …\nQuery:` on queries only. The HTTP server already returns a
//! pooled vector; this client checks width, it does not re-pool.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use serde::Deserialize;

use super::models::{expect_embedding_dim, require_dimensions, resolve_dense_model};
use super::DenseEncoder;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct OllamaEncoder {
    name: String,
    base_url: String,
    score_threshold: Option<f32>,
    dim: usize,
}

impl OllamaEncoder {
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        let name = model.into();
        let dim = resolve_dense_model(&name)
            .map(|s| s.dimensions)
            .unwrap_or(768);
        Self {
            name,
            base_url: base_url.into(),
            score_threshold: Some(0.3),
            dim,
        }
    }

    pub fn localhost(model: impl Into<String>) -> Self {
        let base =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        Self::new(model, base)
    }

    /// Override width. Known models reject a mismatch (Nemotron 2048,
    /// Qwen3-Embedding-0.6B 1024, Qwen3-Embedding-8B 4096, jina-v3 1024,
    /// NV-Embed-v2 4096, Qwen3-VL-Embedding-2B 2048 / 8B 4096,
    /// gte-Qwen2-7B-instruct 3584).
    pub fn with_dimensions(mut self, dim: usize) -> Result<Self> {
        require_dimensions(&self.name, dim)?;
        self.dim = dim;
        Ok(self)
    }
}

#[derive(Deserialize)]
struct EmbedResp {
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    embeddings: Option<Vec<Vec<f32>>>,
}

#[async_trait]
impl DenseEncoder for OllamaEncoder {
    fn name(&self) -> &str {
        &self.name
    }
    fn encoder_type(&self) -> &str {
        "ollama"
    }
    fn score_threshold(&self) -> Option<f32> {
        self.score_threshold
    }
    fn dimensions(&self) -> usize {
        self.dim
    }
    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::Client::new();
        let mut out = Vec::with_capacity(docs.len());
        for doc in docs {
            let url = format!("{}/api/embeddings", self.base_url.trim_end_matches('/'));
            let body = serde_json::json!({"model": self.name, "prompt": doc});
            let resp = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Http(e.to_string()))?;
            if !resp.status().is_success() {
                // newer ollama: /api/embed
                let url2 = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
                let body2 = serde_json::json!({"model": self.name, "input": doc});
                let resp2 = client
                    .post(url2)
                    .json(&body2)
                    .send()
                    .await
                    .map_err(|e| Error::Http(e.to_string()))?;
                if !resp2.status().is_success() {
                    return Err(Error::Http(format!(
                        "ollama embed failed: {}",
                        resp2.status()
                    )));
                }
                let parsed: EmbedResp =
                    resp2.json().await.map_err(|e| Error::Http(e.to_string()))?;
                out.push(first_embedding(parsed, &self.name, self.dim)?);
            } else {
                let parsed: EmbedResp =
                    resp.json().await.map_err(|e| Error::Http(e.to_string()))?;
                out.push(first_embedding(parsed, &self.name, self.dim)?);
            }
        }
        Ok(out)
    }
}

fn first_embedding(parsed: EmbedResp, model: &str, expected_dim: usize) -> Result<Vec<f32>> {
    let emb = if let Some(e) = parsed.embedding {
        e
    } else if let Some(mut es) = parsed.embeddings {
        if es.is_empty() {
            return Err(Error::Http("ollama response missing embedding".into()));
        }
        es.remove(0)
    } else {
        return Err(Error::Http("ollama response missing embedding".into()));
    };
    expect_embedding_dim(model, expected_dim, &emb)?;
    Ok(emb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{
        DenseEncoder, GTE_QWEN2_7B_INSTRUCT_DIM, GTE_QWEN2_7B_INSTRUCT_ID,
        GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG, JINA_EMBEDDINGS_V3_DIM, JINA_EMBEDDINGS_V3_ID,
        JINA_EMBEDDINGS_V3_OLLAMA_TAG, NEMOTRON_3_EMBED_1B_DIM, NEMOTRON_3_EMBED_1B_ID,
        NEMOTRON_3_EMBED_1B_OLLAMA_TAG, NV_EMBED_V2_DIM, NV_EMBED_V2_ID, QWEN3_EMBEDDING_0_6B_DIM,
        QWEN3_EMBEDDING_0_6B_ID, QWEN3_EMBEDDING_0_6B_OLLAMA_TAG, QWEN3_EMBEDDING_8B_DIM,
        QWEN3_EMBEDDING_8B_ID, QWEN3_EMBEDDING_8B_OLLAMA_TAG, QWEN3_VL_EMBEDDING_2B_DIM,
        QWEN3_VL_EMBEDDING_2B_ID, QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG, QWEN3_VL_EMBEDDING_8B_DIM,
        QWEN3_VL_EMBEDDING_8B_ID,
    };

    #[test]
    fn nemotron_http_tag_is_2048() {
        let e = OllamaEncoder::new(NEMOTRON_3_EMBED_1B_OLLAMA_TAG, "http://127.0.0.1:9");
        assert_eq!(e.name(), NEMOTRON_3_EMBED_1B_OLLAMA_TAG);
        assert_eq!(e.dimensions(), NEMOTRON_3_EMBED_1B_DIM);
        let e = OllamaEncoder::new(NEMOTRON_3_EMBED_1B_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), 2048);
    }

    #[test]
    fn nemotron_rejects_mismatched_dim() {
        let err = OllamaEncoder::new(NEMOTRON_3_EMBED_1B_OLLAMA_TAG, "http://127.0.0.1:9")
            .with_dimensions(768)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 768);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mock_http_body_wrong_width_fails() {
        let parsed = EmbedResp {
            embedding: Some(vec![0.1, 0.2, 0.3]),
            embeddings: None,
        };
        let err = first_embedding(parsed, NEMOTRON_3_EMBED_1B_ID, 2048).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mock_http_body_2048_ok() {
        let parsed = EmbedResp {
            embedding: Some(vec![0.0; 2048]),
            embeddings: None,
        };
        let v = first_embedding(parsed, NEMOTRON_3_EMBED_1B_ID, 2048).unwrap();
        assert_eq!(v.len(), 2048);
    }

    #[test]
    fn qwen3_http_tags_are_native_width() {
        let e = OllamaEncoder::new(QWEN3_EMBEDDING_0_6B_OLLAMA_TAG, "http://127.0.0.1:9");
        assert_eq!(e.name(), QWEN3_EMBEDDING_0_6B_OLLAMA_TAG);
        assert_eq!(e.dimensions(), QWEN3_EMBEDDING_0_6B_DIM);
        let e = OllamaEncoder::new(QWEN3_EMBEDDING_0_6B_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), 1024);
        let e = OllamaEncoder::new(QWEN3_EMBEDDING_8B_OLLAMA_TAG, "http://127.0.0.1:9");
        assert_eq!(e.name(), QWEN3_EMBEDDING_8B_OLLAMA_TAG);
        assert_eq!(e.dimensions(), QWEN3_EMBEDDING_8B_DIM);
        let e = OllamaEncoder::new(QWEN3_EMBEDDING_8B_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), 4096);
    }

    #[test]
    fn qwen3_rejects_mismatched_dim() {
        let err = OllamaEncoder::new(QWEN3_EMBEDDING_0_6B_OLLAMA_TAG, "http://127.0.0.1:9")
            .with_dimensions(768)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 768);
            }
            other => panic!("unexpected error: {other}"),
        }
        let err = OllamaEncoder::new(QWEN3_EMBEDDING_8B_OLLAMA_TAG, "http://127.0.0.1:9")
            .with_dimensions(1024)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 4096);
                assert_eq!(got, 1024);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mock_http_body_qwen3_wrong_width_fails() {
        let parsed = EmbedResp {
            embedding: Some(vec![0.1, 0.2, 0.3]),
            embeddings: None,
        };
        let err = first_embedding(parsed, QWEN3_EMBEDDING_0_6B_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mock_http_body_qwen3_native_ok() {
        let parsed = EmbedResp {
            embedding: Some(vec![0.0; 1024]),
            embeddings: None,
        };
        let v = first_embedding(parsed, QWEN3_EMBEDDING_0_6B_ID, 1024).unwrap();
        assert_eq!(v.len(), 1024);
        let parsed = EmbedResp {
            embedding: Some(vec![0.0; 4096]),
            embeddings: None,
        };
        let v = first_embedding(parsed, QWEN3_EMBEDDING_8B_ID, 4096).unwrap();
        assert_eq!(v.len(), 4096);
    }

    #[test]
    fn jina_v3_http_tag_is_1024() {
        let e = OllamaEncoder::new(JINA_EMBEDDINGS_V3_OLLAMA_TAG, "http://127.0.0.1:9");
        assert_eq!(e.name(), JINA_EMBEDDINGS_V3_OLLAMA_TAG);
        assert_eq!(e.dimensions(), JINA_EMBEDDINGS_V3_DIM);
        let e = OllamaEncoder::new(JINA_EMBEDDINGS_V3_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), 1024);
        let err = OllamaEncoder::new(JINA_EMBEDDINGS_V3_OLLAMA_TAG, "http://127.0.0.1:9")
            .with_dimensions(8192)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 8192);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn nv_embed_v2_http_id_is_4096() {
        let e = OllamaEncoder::new(NV_EMBED_V2_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), NV_EMBED_V2_DIM);
        let err = OllamaEncoder::new(NV_EMBED_V2_ID, "http://127.0.0.1:9")
            .with_dimensions(768)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 4096);
                assert_eq!(got, 768);
            }
            other => panic!("unexpected error: {other}"),
        }
        let parsed = EmbedResp {
            embedding: Some(vec![0.0; 4096]),
            embeddings: None,
        };
        let v = first_embedding(parsed, NV_EMBED_V2_ID, 4096).unwrap();
        assert_eq!(v.len(), 4096);
    }

    #[test]
    fn qwen3_vl_http_tags_are_native_width() {
        let e = OllamaEncoder::new(QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG, "http://127.0.0.1:9");
        assert_eq!(e.name(), QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG);
        assert_eq!(e.dimensions(), QWEN3_VL_EMBEDDING_2B_DIM);
        let e = OllamaEncoder::new(QWEN3_VL_EMBEDDING_2B_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), 2048);
        let e = OllamaEncoder::new(QWEN3_VL_EMBEDDING_8B_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), QWEN3_VL_EMBEDDING_8B_DIM);
        let err = OllamaEncoder::new(QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG, "http://127.0.0.1:9")
            .with_dimensions(4096)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 4096);
            }
            other => panic!("unexpected error: {other}"),
        }
        let parsed = EmbedResp {
            embedding: Some(vec![0.0; 2048]),
            embeddings: None,
        };
        let v = first_embedding(parsed, QWEN3_VL_EMBEDDING_2B_ID, 2048).unwrap();
        assert_eq!(v.len(), 2048);
    }

    #[test]
    fn gte_qwen2_http_tag_is_3584() {
        let e = OllamaEncoder::new(GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG, "http://127.0.0.1:9");
        assert_eq!(e.name(), GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG);
        assert_eq!(e.dimensions(), GTE_QWEN2_7B_INSTRUCT_DIM);
        let e = OllamaEncoder::new(GTE_QWEN2_7B_INSTRUCT_ID, "http://127.0.0.1:9");
        assert_eq!(e.dimensions(), 3584);
        let err = OllamaEncoder::new(GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG, "http://127.0.0.1:9")
            .with_dimensions(4096)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 3584);
                assert_eq!(got, 4096);
            }
            other => panic!("unexpected error: {other}"),
        }
        let parsed = EmbedResp {
            embedding: Some(vec![0.0; 3584]),
            embeddings: None,
        };
        let v = first_embedding(parsed, GTE_QWEN2_7B_INSTRUCT_ID, 3584).unwrap();
        assert_eq!(v.len(), 3584);
    }
}
