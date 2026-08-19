//! In-memory dense index — port of semantic_router/index/local.py
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use async_trait::async_trait;
use ndarray::{Array1, Array2};

use super::Index;
use crate::error::{Error, Result};
use crate::linear::{similarity_matrix, top_scores};
use crate::schema::{SparseEmbedding, Utterance, UtteranceRecord};

#[derive(Debug, Clone, Default)]
pub struct LocalIndex {
    records: Vec<UtteranceRecord>,
}

impl LocalIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Index for LocalIndex {
    fn index_type(&self) -> &'static str {
        "local"
    }
    fn is_ready(&self) -> bool {
        !self.records.is_empty()
    }
    fn len(&self) -> usize {
        self.records.len()
    }

    async fn add(&mut self, records: Vec<UtteranceRecord>) -> Result<()> {
        self.records.extend(records);
        Ok(())
    }

    async fn query(
        &self,
        vector: &[f32],
        top_k: usize,
        route_filter: Option<&[String]>,
        _sparse: Option<&SparseEmbedding>,
    ) -> Result<(Vec<f32>, Vec<String>)> {
        if self.records.is_empty() {
            return Err(Error::IndexNotReady);
        }
        let filtered: Vec<&UtteranceRecord> = match route_filter {
            Some(filter) => self
                .records
                .iter()
                .filter(|r| filter.iter().any(|f| f == &r.route))
                .collect(),
            None => self.records.iter().collect(),
        };
        if filtered.is_empty() {
            return Err(Error::EmptyFilter);
        }
        let dim = vector.len();
        let mut data = Vec::with_capacity(filtered.len() * dim);
        for r in &filtered {
            if r.embedding.len() != dim {
                return Err(Error::msg("embedding dimension mismatch"));
            }
            data.extend_from_slice(&r.embedding);
        }
        let index = Array2::from_shape_vec((filtered.len(), dim), data)
            .map_err(|e| Error::msg(e.to_string()))?;
        let xq = Array1::from(vector.to_vec());
        let sim = similarity_matrix(&xq, &index);
        let (scores, idx) = top_scores(&sim, top_k);
        let routes = idx.iter().map(|&i| filtered[i].route.clone()).collect();
        Ok((scores, routes))
    }

    async fn delete_route(&mut self, route_name: &str) -> Result<()> {
        self.records.retain(|r| r.route != route_name);
        Ok(())
    }

    async fn get_utterances(&self, include_metadata: bool) -> Result<Vec<Utterance>> {
        Ok(self
            .records
            .iter()
            .map(|r| {
                let mut u = Utterance::new(&r.route, &r.utterance);
                if include_metadata {
                    u.metadata = r.metadata.clone();
                    u.function_schemas = r.function_schemas.clone();
                }
                u
            })
            .collect())
    }

    async fn clear(&mut self) -> Result<()> {
        self.records.clear();
        Ok(())
    }
}
