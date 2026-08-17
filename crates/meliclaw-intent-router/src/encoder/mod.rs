//! Encoders — dense (hash / HTTP / ONNX metadata) and optional sparse.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

mod hash;
mod onnx;

#[cfg(feature = "hybrid")]
mod bm25;
#[cfg(feature = "hybrid")]
mod tfidf;

#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "openai")]
mod openai;

use async_trait::async_trait;

use crate::error::Result;
use crate::schema::SparseEmbedding;

pub use hash::HashDenseEncoder;
pub use onnx::OnnxEncoder;

#[cfg(feature = "hybrid")]
pub use bm25::Bm25Encoder;
#[cfg(feature = "ollama")]
pub use ollama::OllamaEncoder;
#[cfg(feature = "openai")]
pub use openai::OpenAiEncoder;
#[cfg(feature = "hybrid")]
pub use tfidf::TfidfEncoder;

#[async_trait]
pub trait DenseEncoder: Send + Sync {
    fn name(&self) -> &str;
    fn encoder_type(&self) -> &str;
    fn score_threshold(&self) -> Option<f32>;
    fn dimensions(&self) -> usize;

    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>>;

    async fn encode_queries(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.encode(docs).await
    }

    async fn encode_documents(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.encode(docs).await
    }
}

#[async_trait]
pub trait SparseEncoder: Send + Sync {
    fn name(&self) -> &str;
    async fn encode_queries(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>>;
    async fn encode_documents(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>>;
}
