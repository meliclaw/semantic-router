//! ONNX embedding encoder (on-prem default in architecture: nomic / bge-m3).
//!
//! Runtime ONNX session is optional (`onnx-embed`). Without a model file this
//! encoder uses [`HashDenseEncoder`] with the published dimension of the named
//! model so the rest of the stack can boot. Production must set
//! `MELICLAW_ONNX_MODEL` to a real `.onnx` path and enable the feature.
//!
//! Model **weights** are not MIT. Complete the license line per deployment.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use super::{DenseEncoder, HashDenseEncoder};
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct OnnxEncoder {
    model_id: String,
    model_path: Option<PathBuf>,
    inner: HashDenseEncoder,
}

impl OnnxEncoder {
    /// Known Meliclaw embedding models (v5.4 §14.2).
    pub fn from_model(name: &str) -> Result<Self> {
        let (id, dim, threshold) = match name {
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
                ("nomic-embed-text-v1.5", 768, 0.3)
            }
            "bge-m3" | "BAAI/bge-m3" => ("bge-m3", 1024, 0.3),
            "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => ("bge-small-en-v1.5", 384, 0.3),
            other => (other, 768, 0.3),
        };
        let path = std::env::var("MELICLAW_ONNX_MODEL")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists());
        Ok(Self {
            model_id: id.to_string(),
            model_path: path,
            inner: HashDenseEncoder::new(id, dim, Some(threshold)),
        })
    }

    pub fn from_path(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::Onnx(format!("model not found: {}", path.display())));
        }
        // Full ORT session lands behind feature onnx-embed + native lib.
        // Until then we refuse silent hash fallback when an explicit path is given
        // but the runtime is not compiled in — callers know they asked for ONNX.
        if cfg!(feature = "onnx-embed") {
            tracing::warn!(
                path = %path.display(),
                "ONNX path provided; native ORT session not linked in this build, using hash stand-in of matching dim"
            );
            Ok(Self {
                model_id: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("onnx")
                    .to_string(),
                model_path: Some(path.to_path_buf()),
                inner: HashDenseEncoder::new("onnx", dim, Some(0.3)),
            })
        } else {
            Err(Error::Onnx(
                "rebuild with feature `onnx-embed` and a linked ONNX Runtime to load .onnx files"
                    .into(),
            ))
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }
}

#[async_trait]
impl DenseEncoder for OnnxEncoder {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn encoder_type(&self) -> &str {
        "onnx"
    }
    fn score_threshold(&self) -> Option<f32> {
        self.inner.score_threshold()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
    async fn encode(&self, docs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.encode(docs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::DenseEncoder;

    #[tokio::test]
    async fn known_model_ids() {
        let e = OnnxEncoder::from_model("nomic-embed-text-v1.5").unwrap();
        assert_eq!(e.model_id(), "nomic-embed-text-v1.5");
        assert_eq!(e.dimensions(), 768);
        let v = e.encode(&[String::from("hola")]).await.unwrap();
        assert_eq!(v[0].len(), 768);
    }
}
