//! Ollama / LM Studio / vLLM embeddings. Feature `ollama`.
//!
//! Nemotron-3-Embed-1B is served over this HTTP API (Ollama tag
//! `nemotron-3-embed-1b`, or Hugging Face id `nvidia/Nemotron-3-Embed-1B-BF16`
//! when the proxy forwards that name). Native width is 2048.
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

    /// Override width. Known models (including Nemotron at 2048) reject a mismatch.
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
        DenseEncoder, NEMOTRON_3_EMBED_1B_DIM, NEMOTRON_3_EMBED_1B_ID,
        NEMOTRON_3_EMBED_1B_OLLAMA_TAG,
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
}
