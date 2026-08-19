//! Hybrid dense+sparse in-memory index — port of index/hybrid_local.py
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use async_trait::async_trait;
use ndarray::{Array1, Array2};

use super::Index;
use crate::error::{Error, Result};
use crate::linear::similarity_matrix;
use crate::schema::{SparseEmbedding, Utterance, UtteranceRecord};

#[derive(Debug, Clone, Default)]
pub struct HybridLocalIndex {
    records: Vec<UtteranceRecord>,
}

impl HybridLocalIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Index for HybridLocalIndex {
    fn index_type(&self) -> &'static str {
        "hybrid_local"
    }
    fn is_ready(&self) -> bool {
        !self.records.is_empty()
    }
    fn len(&self) -> usize {
        self.records.len()
    }

    async fn add(&mut self, records: Vec<UtteranceRecord>) -> Result<()> {
        if records.iter().any(|r| r.sparse.is_none()) {
            return Err(Error::MissingSparse);
        }
        self.records.extend(records);
        Ok(())
    }

    async fn query(
        &self,
        vector: &[f32],
        top_k: usize,
        route_filter: Option<&[String]>,
        sparse: Option<&SparseEmbedding>,
    ) -> Result<(Vec<f32>, Vec<String>)> {
        if route_filter.is_some() {
            return Err(Error::HybridRouteFilter);
        }
        let sparse = sparse.ok_or(Error::MissingSparse)?;
        if self.records.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let dim = vector.len();
        let mut data = Vec::with_capacity(self.records.len() * dim);
        for r in &self.records {
            data.extend_from_slice(&r.embedding);
        }
        let index = Array2::from_shape_vec((self.records.len(), dim), data)
            .map_err(|e| Error::msg(e.to_string()))?;
        let xq = Array1::from(vector.to_vec());
        let sim_d = similarity_matrix(&xq, &index);
        let mut total = Vec::with_capacity(self.records.len());
        for (i, r) in self.records.iter().enumerate() {
            let sim_s = r.sparse.as_ref().map(|s| s.dot(sparse)).unwrap_or(0.0);
            total.push(sim_d[i] + sim_s);
        }
        let k = top_k.min(total.len());
        let mut idx: Vec<usize> = (0..total.len()).collect();
        idx.sort_by(|&a, &b| {
            total[b]
                .partial_cmp(&total[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        // Python returns the selected set unsorted relative to original argpartition;
        // we return descending by total score for hybrid (more useful). Tests check membership.
        let scores = idx.iter().map(|&i| total[i]).collect();
        let routes = idx.iter().map(|&i| self.records[i].route.clone()).collect();
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
