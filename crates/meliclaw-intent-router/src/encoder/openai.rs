//! OpenAI embeddings HTTP encoder. Feature `openai`.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use serde::Deserialize;

use super::DenseEncoder;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct OpenAiEncoder {
    name: String,
    api_key: String,
    base_url: String,
    score_threshold: Option<f32>,
    dim: usize,
}

impl OpenAiEncoder {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            name: model.into(),
            api_key: api_key.into(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            score_threshold: Some(0.3),
            dim: 1536,
        }
    }
}

#[derive(Deserialize)]
struct EmbResponse {
    data: Vec<EmbData>,
}
#[derive(Deserialize)]
struct EmbData {
    embedding: Vec<f32>,
}

#[async_trait]
impl DenseEncoder for OpenAiEncoder {
    fn name(&self) -> &str {
        &self.name
    }
    fn encoder_type(&self) -> &str {
        "openai"
    }
    fn score_threshold(&self) -> Option<f32> {
        self.score_threshold
    }
    fn dimensions(&self) -> usize {
        self.dim
    }
    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::Client::new();
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.name,
            "input": docs,
        });
        let resp = client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Http(format!(
                "openai embeddings {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            )));
        }
        let parsed: EmbResponse = resp.json().await.map_err(|e| Error::Http(e.to_string()))?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}
