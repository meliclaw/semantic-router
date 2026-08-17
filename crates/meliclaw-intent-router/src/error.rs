//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.
//! Licensed under the MIT License.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("index is not ready")]
    IndexNotReady,
    #[error("either text or vector must be provided")]
    MissingQuery,
    #[error("unsupported aggregation: {0}")]
    BadAggregation(String),
    #[error("no routes matched the filter")]
    EmptyFilter,
    #[error("encoder not fitted")]
    EncoderNotFitted,
    #[error("sparse vector required for hybrid index")]
    MissingSparse,
    #[error("route filter is not supported for HybridLocalIndex")]
    HybridRouteFilter,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("http: {0}")]
    Http(String),
    #[error("database: {0}")]
    Database(String),
    #[error("sync error: local and remote utterances differ")]
    SyncConflict,
    #[error("onnx: {0}")]
    Onnx(String),
}

impl Error {
    pub fn msg(m: impl Into<String>) -> Self {
        Self::Message(m.into())
    }
}

#[cfg(feature = "openai")]
impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e.to_string())
    }
}
