//! Encoders — dense (hash / HTTP / ONNX metadata) and optional sparse.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

mod hash;
mod models;
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
pub use models::{
    expect_embedding_dim, pool_last_hidden_state, require_dimensions, resolve_dense_model,
    DenseModelSpec, EmbeddingPooling, GTE_QWEN2_7B_INSTRUCT_DIM,
    GTE_QWEN2_7B_INSTRUCT_GGUF_Q4_K_M_FILE, GTE_QWEN2_7B_INSTRUCT_GGUF_REPO,
    GTE_QWEN2_7B_INSTRUCT_ID, GTE_QWEN2_7B_INSTRUCT_OLLAMA_COMMUNITY_Q4_K_M,
    GTE_QWEN2_7B_INSTRUCT_OLLAMA_TAG, JINA_EMBEDDINGS_V3_DIM, JINA_EMBEDDINGS_V3_GGUF_Q4_K_M_FILE,
    JINA_EMBEDDINGS_V3_GGUF_REPO, JINA_EMBEDDINGS_V3_ID, JINA_EMBEDDINGS_V3_OLLAMA_TAG,
    JINA_EMBEDDINGS_V3_ONNX_FILE, JINA_EMBEDDINGS_V3_ONNX_FP16_FILE, NEMOTRON_3_EMBED_1B_DIM,
    NEMOTRON_3_EMBED_1B_ID, NEMOTRON_3_EMBED_1B_OLLAMA_TAG, NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
    NEMOTRON_3_EMBED_1B_ONNX_FILE, NEMOTRON_3_EMBED_1B_ONNX_FP16_FILE, NV_EMBED_V2_DIM,
    NV_EMBED_V2_ID, QWEN3_EMBEDDING_0_6B_DIM, QWEN3_EMBEDDING_0_6B_ID,
    QWEN3_EMBEDDING_0_6B_OLLAMA_TAG, QWEN3_EMBEDDING_0_6B_ONNX_EXPORT,
    QWEN3_EMBEDDING_0_6B_ONNX_FILE, QWEN3_EMBEDDING_8B_DIM, QWEN3_EMBEDDING_8B_ID,
    QWEN3_EMBEDDING_8B_OLLAMA_TAG, QWEN3_EMBEDDING_8B_ONNX_EXPORT, QWEN3_EMBEDDING_8B_ONNX_FILE,
    QWEN3_VL_EMBEDDING_2B_DIM, QWEN3_VL_EMBEDDING_2B_GGUF_Q4_K_M_FILE,
    QWEN3_VL_EMBEDDING_2B_GGUF_REPO, QWEN3_VL_EMBEDDING_2B_ID, QWEN3_VL_EMBEDDING_2B_OLLAMA_TAG,
    QWEN3_VL_EMBEDDING_8B_DIM, QWEN3_VL_EMBEDDING_8B_GGUF_Q4_K_M_FILE,
    QWEN3_VL_EMBEDDING_8B_GGUF_REPO, QWEN3_VL_EMBEDDING_8B_ID,
};
pub use onnx::OnnxEncoder;

#[cfg(feature = "hybrid")]
pub use bm25::Bm25Encoder;
#[cfg(feature = "ollama")]
pub use ollama::OllamaEncoder;
#[cfg(feature = "openai")]
pub use openai::{embeddings_request_body, OpenAiEncoder};
#[cfg(feature = "hybrid")]
pub use tfidf::TfidfEncoder;

#[async_trait]
pub trait DenseEncoder: Send + Sync {
    fn name(&self) -> &str;
    fn encoder_type(&self) -> &str;
    fn score_threshold(&self) -> Option<f32>;
    fn dimensions(&self) -> usize;

    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Encode queries. Capa 1 utterances are query-like: this must stay
    /// identical to [`encode`] / [`encode_documents`]. Do not apply Qwen
    /// retrieval `Instruct: …\nQuery:` on queries only.
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
