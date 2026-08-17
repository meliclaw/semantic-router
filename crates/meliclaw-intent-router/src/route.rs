//! Route definition — static routes in v1 (no LLM function extraction).
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::schema::RouteChoice;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Route {
    pub name: String,
    pub utterances: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub function_schemas: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Route {
    pub fn new(name: impl Into<String>, utterances: Vec<impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            utterances: utterances.into_iter().map(Into::into).collect(),
            description: None,
            function_schemas: None,
            score_threshold: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Static route: name only. Dynamic function_call is v1 non-goal.
    pub fn choose(&self, score: f32) -> RouteChoice {
        RouteChoice::named(&self.name, score)
    }
}
