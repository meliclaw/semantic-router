//! OpenAI embeddings HTTP encoder. Feature `openai`.
//!
//! Also used as a LocalAI client: LocalAI's `/v1/embeddings` is OpenAI-compatible.
//! Text requests keep `{ "model", "input": [strings] }`. Multimodal VL probes use
//! [`OpenAiEncoder::encode_with_images`], which puts text on `input.text` and
//! `data:image/...;base64,...` URLs on `input.images` — a documented LocalAI-facing
//! payload. Native LocalAI embeddings are typically text-only; VL backends must
//! accept that object form. Do not send images through [`DenseEncoder::encode`].
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::models::{expect_embedding_dim, require_dimensions, resolve_dense_model};
use super::DenseEncoder;
use crate::error::{Error, Result};

const DEFAULT_OPENAI_BASE: &str = "https://api.openai.com/v1";
const DEFAULT_LOCALAI_BASE: &str = "http://127.0.0.1:8080/v1";
const DEFAULT_OPENAI_DIM: usize = 1536;

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
        let name = model.into();
        let (dim, threshold) = resolve_dense_model(&name)
            .map(|s| (s.dimensions, s.score_threshold))
            .unwrap_or((DEFAULT_OPENAI_DIM, 0.3));
        Self {
            name,
            api_key: api_key.into(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_OPENAI_BASE.into()),
            score_threshold: Some(threshold),
            dim,
        }
    }

    /// LocalAI OpenAI-compatible embeddings (`POST {base}/embeddings`).
    /// Default base is `http://127.0.0.1:8080/v1` (or `LOCALAI_BASE_URL`).
    pub fn localai(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let base =
            std::env::var("LOCALAI_BASE_URL").unwrap_or_else(|_| DEFAULT_LOCALAI_BASE.into());
        Self::new(model, api_key).with_base_url(base)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override width. Known catalog models reject a mismatch.
    pub fn with_dimensions(mut self, dim: usize) -> Result<Self> {
        require_dimensions(&self.name, dim)?;
        self.dim = dim;
        Ok(self)
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// One multimodal embedding: text plus image data URLs.
    ///
    /// Images must already be `data:<mime>;base64,...`. Empty `image_data_urls`
    /// falls back to the text-only OpenAI body so the service path stays unchanged.
    pub async fn encode_with_images(
        &self,
        texts: &[String],
        image_data_urls: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        if image_data_urls.is_empty() && texts.iter().all(|t| t.is_empty()) {
            return Err(Error::msg(
                "encode_with_images requires text or at least one image",
            ));
        }
        self.post_embeddings(&embeddings_request_body(&self.name, texts, image_data_urls))
            .await
    }

    async fn post_embeddings(&self, body: &Value) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::Client::new();
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let resp = client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(body)
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
        embeddings_from_response(parsed, &self.name, self.dim)
    }
}

/// Text-only: `{ "model", "input": [strings] }`.
/// With images: `{ "model", "input": { "text"?, "images": ["data:image/...;base64,..."] } }`.
pub fn embeddings_request_body(model: &str, texts: &[String], image_data_urls: &[String]) -> Value {
    if image_data_urls.is_empty() {
        return json!({ "model": model, "input": texts });
    }
    let text = texts
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let mut input = serde_json::Map::new();
    if !text.is_empty() {
        input.insert("text".into(), json!(text));
    }
    input.insert("images".into(), json!(image_data_urls));
    json!({ "model": model, "input": input })
}

#[derive(Deserialize)]
struct EmbResponse {
    data: Vec<EmbData>,
}
#[derive(Deserialize)]
struct EmbData {
    embedding: Vec<f32>,
}

fn embeddings_from_response(
    parsed: EmbResponse,
    model: &str,
    expected_dim: usize,
) -> Result<Vec<Vec<f32>>> {
    if parsed.data.is_empty() {
        return Err(Error::Http(
            "openai embeddings response missing data".into(),
        ));
    }
    let mut out = Vec::with_capacity(parsed.data.len());
    for d in parsed.data {
        expect_embedding_dim(model, expected_dim, &d.embedding)?;
        out.push(d.embedding);
    }
    Ok(out)
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
        self.post_embeddings(&embeddings_request_body(&self.name, docs, &[]))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{
        DenseEncoder, GTE_QWEN2_7B_INSTRUCT_DIM, GTE_QWEN2_7B_INSTRUCT_ID, NEMOTRON_3_EMBED_1B_DIM,
        NEMOTRON_3_EMBED_1B_ID, NEMOTRON_3_EMBED_1B_OLLAMA_TAG, QWEN3_EMBEDDING_8B_DIM,
        QWEN3_EMBEDDING_8B_ID, QWEN3_VL_EMBEDDING_2B_DIM, QWEN3_VL_EMBEDDING_2B_ID,
        QWEN3_VL_EMBEDDING_8B_DIM, QWEN3_VL_EMBEDDING_8B_ID,
    };

    #[test]
    fn catalog_models_use_native_width() {
        assert_eq!(
            OpenAiEncoder::new("nomic-embed-text-v1.5", "sk").dimensions(),
            768
        );
        assert_eq!(OpenAiEncoder::new("bge-m3", "sk").dimensions(), 1024);
        assert_eq!(
            OpenAiEncoder::new("bge-small-en-v1.5", "sk").dimensions(),
            384
        );
        assert_eq!(
            OpenAiEncoder::new(NEMOTRON_3_EMBED_1B_ID, "sk").dimensions(),
            NEMOTRON_3_EMBED_1B_DIM
        );
        assert_eq!(
            OpenAiEncoder::new(NEMOTRON_3_EMBED_1B_OLLAMA_TAG, "sk").dimensions(),
            2048
        );
        assert_eq!(
            OpenAiEncoder::new(QWEN3_EMBEDDING_8B_ID, "sk").dimensions(),
            QWEN3_EMBEDDING_8B_DIM
        );
        assert_eq!(
            OpenAiEncoder::new(QWEN3_VL_EMBEDDING_2B_ID, "sk").dimensions(),
            QWEN3_VL_EMBEDDING_2B_DIM
        );
        assert_eq!(
            OpenAiEncoder::new(QWEN3_VL_EMBEDDING_8B_ID, "sk").dimensions(),
            QWEN3_VL_EMBEDDING_8B_DIM
        );
        assert_eq!(
            OpenAiEncoder::new(GTE_QWEN2_7B_INSTRUCT_ID, "sk").dimensions(),
            GTE_QWEN2_7B_INSTRUCT_DIM
        );
        assert_eq!(
            OpenAiEncoder::new("gte-qwen2-7b-instruct", "sk").dimensions(),
            3584
        );
    }

    #[test]
    fn unknown_openai_model_keeps_1536() {
        assert_eq!(
            OpenAiEncoder::new("text-embedding-3-small", "sk").dimensions(),
            1536
        );
    }

    #[test]
    fn rejects_mismatched_dim() {
        let err = OpenAiEncoder::new("bge-m3", "sk")
            .with_dimensions(768)
            .unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 768);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn localai_defaults_loopback_v1() {
        let e = OpenAiEncoder::localai("bge-m3", "sk-local");
        assert_eq!(e.dimensions(), 1024);
        assert!(e.base_url().ends_with("/v1"), "{}", e.base_url());
    }

    #[test]
    fn with_base_url_overrides() {
        let e = OpenAiEncoder::new("bge-m3", "sk").with_base_url("http://127.0.0.1:8080/v1");
        assert_eq!(e.base_url(), "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn text_body_is_openai_compatible() {
        let v = embeddings_request_body("bge-m3", &["hello".into()], &[]);
        assert_eq!(v["model"], "bge-m3");
        assert_eq!(v["input"][0], "hello");
        assert!(v["input"].is_array());
    }

    #[test]
    fn multimodal_body_uses_input_object() {
        let v = embeddings_request_body(
            "qwen3-vl-embedding-2b",
            &["caption".into()],
            &["data:image/png;base64,AAA".into()],
        );
        assert_eq!(v["model"], "qwen3-vl-embedding-2b");
        assert_eq!(v["input"]["text"], "caption");
        assert_eq!(v["input"]["images"][0], "data:image/png;base64,AAA");
        assert!(v["input"].is_object());
    }

    #[test]
    fn multimodal_images_only_omits_empty_text() {
        let v = embeddings_request_body(
            "qwen3-vl-embedding-2b",
            &["".into()],
            &["data:image/jpeg;base64,BBB".into()],
        );
        assert!(v["input"].get("text").is_none());
        assert_eq!(v["input"]["images"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn mock_http_body_wrong_width_fails() {
        let parsed = EmbResponse {
            data: vec![EmbData {
                embedding: vec![0.1, 0.2, 0.3],
            }],
        };
        let err = embeddings_from_response(parsed, NEMOTRON_3_EMBED_1B_ID, 2048).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn mock_http_body_native_width_ok() {
        let parsed = EmbResponse {
            data: vec![EmbData {
                embedding: vec![0.0; 2048],
            }],
        };
        let v = embeddings_from_response(parsed, NEMOTRON_3_EMBED_1B_ID, 2048).unwrap();
        assert_eq!(v[0].len(), 2048);
    }

    #[test]
    fn mock_empty_data_is_http_error() {
        let parsed = EmbResponse { data: vec![] };
        let err = embeddings_from_response(parsed, "bge-m3", 1024).unwrap_err();
        match err {
            Error::Http(m) => assert!(m.contains("missing data"), "{m}"),
            other => panic!("unexpected error: {other}"),
        }
    }
}
