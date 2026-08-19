//! Routers: SemanticRouter (dense) and optional HybridRouter.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

mod semantic;

#[cfg(feature = "hybrid")]
mod hybrid;

use serde::{Deserialize, Serialize};

pub use semantic::SemanticRouter;

#[cfg(feature = "hybrid")]
pub use hybrid::HybridRouter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Aggregation {
    Sum,
    Mean,
    Max,
}

impl Aggregation {
    pub fn parse(s: &str) -> crate::error::Result<Self> {
        match s {
            "sum" => Ok(Self::Sum),
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            other => Err(crate::error::Error::BadAggregation(other.into())),
        }
    }

    pub fn apply(self, scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        match self {
            Self::Sum => scores.iter().sum(),
            Self::Mean => scores.iter().sum::<f32>() / scores.len() as f32,
            Self::Max => scores.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RouteRequest {
    pub text: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub simulate_static: bool,
    pub route_filter: Option<Vec<String>>,
    pub limit: Option<usize>,
}

impl RouteRequest {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            limit: Some(1),
            ..Default::default()
        }
    }
}
