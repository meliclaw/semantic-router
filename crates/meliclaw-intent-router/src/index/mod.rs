//! Vector indexes.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

mod local;

#[cfg(feature = "hybrid")]
mod hybrid_local;

#[cfg(feature = "postgres")]
mod postgres;

#[cfg(feature = "qdrant")]
mod qdrant;

use async_trait::async_trait;

use crate::error::Result;
use crate::schema::{SparseEmbedding, Utterance, UtteranceRecord};

pub use local::LocalIndex;

#[cfg(feature = "hybrid")]
pub use hybrid_local::HybridLocalIndex;

#[cfg(feature = "postgres")]
pub use postgres::PostgresIndex;

#[cfg(feature = "qdrant")]
pub use qdrant::QdrantIndex;

#[async_trait]
pub trait Index: Send + Sync {
    fn index_type(&self) -> &'static str;
    fn is_ready(&self) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    async fn add(&mut self, records: Vec<UtteranceRecord>) -> Result<()>;
    async fn query(
        &self,
        vector: &[f32],
        top_k: usize,
        route_filter: Option<&[String]>,
        sparse: Option<&SparseEmbedding>,
    ) -> Result<(Vec<f32>, Vec<String>)>;
    async fn delete_route(&mut self, route_name: &str) -> Result<()>;
    async fn get_utterances(&self, include_metadata: bool) -> Result<Vec<Utterance>>;
    async fn clear(&mut self) -> Result<()>;
}
