//! Schema types — port of semantic_router/schema.py (without aurelio_sdk).
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RouteChoice {
    pub name: Option<String>,
    #[serde(default)]
    pub function_call: Option<Vec<serde_json::Value>>,
    pub similarity_score: Option<f32>,
}

impl RouteChoice {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn named(name: impl Into<String>, score: f32) -> Self {
        Self {
            name: Some(name.into()),
            function_call: None,
            similarity_score: Some(score),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Utterance {
    pub route: String,
    pub utterance: String,
    #[serde(default)]
    pub function_schemas: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default = "space_tag")]
    pub diff_tag: String,
}

fn space_tag() -> String {
    " ".to_string()
}

impl Utterance {
    pub fn new(route: impl Into<String>, utterance: impl Into<String>) -> Self {
        Self {
            route: route.into(),
            utterance: utterance.into(),
            function_schemas: None,
            metadata: BTreeMap::new(),
            diff_tag: " ".into(),
        }
    }

    pub fn to_str(&self, include_metadata: bool) -> String {
        if include_metadata {
            let schemas = self
                .function_schemas
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap_or_default());
            let meta = serde_json::to_string(&self.metadata).unwrap_or_else(|_| "{}".into());
            format!(
                "{}: {} | {:?} | {}",
                self.route, self.utterance, schemas, meta
            )
        } else {
            format!("{}: {}", self.route, self.utterance)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SyncMode {
    Error,
    Remote,
    Local,
    MergeForceRemote,
    MergeForceLocal,
    Merge,
}

impl SyncMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Remote => "remote",
            Self::Local => "local",
            Self::MergeForceRemote => "merge-force-remote",
            Self::MergeForceLocal => "merge-force-local",
            Self::Merge => "merge",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "error" => Some(Self::Error),
            "remote" => Some(Self::Remote),
            "local" => Some(Self::Local),
            "merge-force-remote" => Some(Self::MergeForceRemote),
            "merge-force-local" => Some(Self::MergeForceLocal),
            "merge" => Some(Self::Merge),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SparseEmbedding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseEmbedding {
    pub fn from_dense(vec: &[f32]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in vec.iter().enumerate() {
            if v != 0.0 {
                indices.push(i as u32);
                values.push(v);
            }
        }
        Self { indices, values }
    }

    pub fn to_map(&self) -> BTreeMap<u32, f32> {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .collect()
    }

    pub fn from_map(map: &BTreeMap<u32, f32>) -> Self {
        let (indices, values): (Vec<_>, Vec<_>) = map.iter().map(|(&k, &v)| (k, v)).unzip();
        Self { indices, values }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        let b = other.to_map();
        self.indices
            .iter()
            .zip(self.values.iter())
            .map(|(i, v)| v * b.get(i).copied().unwrap_or(0.0))
            .sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceRecord {
    pub route: String,
    pub utterance: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub sparse: Option<SparseEmbedding>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub function_schemas: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigParameter {
    pub field: String,
    pub value: String,
    pub scope: Option<String>,
}
