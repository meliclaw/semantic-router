//! JSON/YAML RouterConfig — port of routers/base.py RouterConfig.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::error::Result;
use crate::route::Route;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterConfig {
    #[serde(default = "default_encoder_type")]
    pub encoder_type: String,
    #[serde(default)]
    pub encoder_name: Option<String>,
    #[serde(default)]
    pub routes: Vec<Route>,
}

fn default_encoder_type() -> String {
    "hash".into()
}

impl RouterConfig {
    pub fn new(routes: Vec<Route>) -> Self {
        Self {
            encoder_type: "hash".into(),
            encoder_name: Some("hash-dense".into()),
            routes,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let cfg: Self = match ext.as_str() {
            "json" => serde_json::from_str(&text)?,
            "yaml" | "yml" => serde_yaml::from_str(&text)?,
            _ => {
                if text.trim_start().starts_with('{') {
                    serde_json::from_str(&text)?
                } else {
                    serde_yaml::from_str(&text)?
                }
            }
        };
        Ok(cfg)
    }

    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("json")
            .to_ascii_lowercase();
        let text = if ext == "yaml" || ext == "yml" {
            serde_yaml::to_string(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn hash(&self) -> String {
        let payload = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::Route;

    #[test]
    fn roundtrip_yaml() {
        let cfg = RouterConfig::new(vec![Route::new("factual", vec!["NIF de"])]);
        let dir = std::env::temp_dir();
        let path = dir.join("meliclaw-intent-router-test.yaml");
        cfg.to_file(&path).unwrap();
        let loaded = RouterConfig::from_file(&path).unwrap();
        assert_eq!(loaded.routes[0].name, "factual");
        let _ = std::fs::remove_file(&path);
    }
}
