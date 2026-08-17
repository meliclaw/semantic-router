//! Deterministic bag-of-words dense encoder (no torch / no network).
//! Used for tests, golden routing, and as ONNX stand-in until a session is loaded.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::DenseEncoder;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct HashDenseEncoder {
    name: String,
    dim: usize,
    score_threshold: Option<f32>,
}

impl Default for HashDenseEncoder {
    fn default() -> Self {
        Self::new("hash-dense", 256, Some(0.3))
    }
}

impl HashDenseEncoder {
    pub fn new(name: impl Into<String>, dim: usize, score_threshold: Option<f32>) -> Self {
        Self {
            name: name.into(),
            dim: dim.max(8),
            score_threshold,
        }
    }

    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for token in tokenize(text) {
            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            let h = hasher.finish();
            let idx = (h as usize) % self.dim;
            let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        v
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[async_trait]
impl DenseEncoder for HashDenseEncoder {
    fn name(&self) -> &str {
        &self.name
    }
    fn encoder_type(&self) -> &str {
        "hash"
    }
    fn score_threshold(&self) -> Option<f32> {
        self.score_threshold
    }
    fn dimensions(&self) -> usize {
        self.dim
    }
    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(docs.iter().map(|d| self.embed_one(d)).collect())
    }
}
