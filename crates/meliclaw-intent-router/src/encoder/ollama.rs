//! Ollama / LM Studio embeddings. Feature `ollama`.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use serde::Deserialize;

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
        Self {
            name: model.into(),
            base_url: base_url.into(),
            score_threshold: Some(0.3),
            dim: 768,
        }
    }

    pub fn localhost(model: impl Into<String>) -> Self {
        let base =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        Self::new(model, base)
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
                out.push(first_embedding(parsed)?);
            } else {
                let parsed: EmbedResp =
                    resp.json().await.map_err(|e| Error::Http(e.to_string()))?;
                out.push(first_embedding(parsed)?);
            }
        }
        Ok(out)
    }
}

fn first_embedding(parsed: EmbedResp) -> Result<Vec<f32>> {
    if let Some(e) = parsed.embedding {
        return Ok(e);
    }
    if let Some(mut es) = parsed.embeddings {
        if !es.is_empty() {
            return Ok(es.remove(0));
        }
    }
    Err(Error::Http("ollama response missing embedding".into()))
}
