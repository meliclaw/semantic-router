//! Meliclaw Intent Router — Capa 1 (Intent Classifier).
//!
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.
//! Licensed under the MIT License. Algorithm ported from
//! https://github.com/aurelio-labs/semantic-router

pub mod config;
pub mod encoder;
pub mod error;
pub mod index;
pub mod linear;
pub mod memory_routes;
pub mod route;
pub mod router;
pub mod schema;
pub mod sync;

pub use config::RouterConfig;
#[cfg(feature = "ollama")]
pub use encoder::OllamaEncoder;
#[cfg(feature = "openai")]
pub use encoder::OpenAiEncoder;
pub use encoder::{
    DenseEncoder, HashDenseEncoder, OnnxEncoder, SparseEncoder, NEMOTRON_3_EMBED_1B_DIM,
    NEMOTRON_3_EMBED_1B_ID, NEMOTRON_3_EMBED_1B_OLLAMA_TAG, NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
    NEMOTRON_3_EMBED_1B_ONNX_FILE, NEMOTRON_3_EMBED_1B_ONNX_FP16_FILE,
};
pub use error::{Error, Result};
#[cfg(feature = "postgres")]
pub use index::PostgresIndex;
#[cfg(feature = "qdrant")]
pub use index::QdrantIndex;
pub use index::{Index, LocalIndex};
pub use memory_routes::memory_intent_routes;
pub use route::Route;
pub use router::{Aggregation, RouteRequest, SemanticRouter};
pub use schema::{RouteChoice, SparseEmbedding, SyncMode, Utterance};

#[cfg(feature = "hybrid")]
pub use encoder::{Bm25Encoder, TfidfEncoder};
#[cfg(feature = "hybrid")]
pub use index::HybridLocalIndex;
#[cfg(feature = "hybrid")]
pub use router::HybridRouter;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const UPSTREAM_VERSION: &str = "0.1.16";
