//! ONNX embedding encoder (on-prem default in architecture: nomic / bge-m3).
//!
//! Runtime ONNX session is optional (`onnx-embed`). Without a model file this
//! encoder uses [`HashDenseEncoder`] with the published dimension of the named
//! model so the rest of the stack can boot. Production must set
//! `MELICLAW_ONNX_MODEL` to a real `.onnx` path and enable the feature.
//!
//! NVIDIA Nemotron-3-Embed-1B is a first-class option (2048-d, GPU). There is
//! no NVIDIA-hosted ONNX URL; operators point `MELICLAW_ONNX_MODEL` at a local
//! graph (`model.onnx` or `fp16/model.onnx` from the community export
//! `kzzalews/Nemotron-3-Embed-1B-BF16-onnx`) after checksumming against
//! `nvidia/Nemotron-3-Embed-1B-BF16`.
//!
//! Model **weights** are not MIT. Complete the license line per deployment.
//! Modified by Meliclaw, 2026.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use super::models::{require_dimensions, resolve_dense_model};
use super::{DenseEncoder, HashDenseEncoder};
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct OnnxEncoder {
    model_id: String,
    model_path: Option<PathBuf>,
    inner: HashDenseEncoder,
}

impl OnnxEncoder {
    /// Known Meliclaw embedding models (v5.4 §14.2). Unknown names keep dim 768.
    pub fn from_model(name: &str) -> Result<Self> {
        let (id, dim, threshold) = match resolve_dense_model(name) {
            Some(spec) => (spec.id, spec.dimensions, spec.score_threshold),
            None => (name, 768, 0.3),
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

    /// Same as [`from_model`], but reject a known model when `dim` is wrong.
    ///
    /// Nemotron-3-Embed-1B must be 2048. Unknown names use `dim` as-is.
    pub fn from_model_with_dimensions(name: &str, dim: usize) -> Result<Self> {
        require_dimensions(name, dim)?;
        let mut enc = Self::from_model(name)?;
        if enc.dimensions() != dim {
            enc.inner = HashDenseEncoder::new(enc.model_id.clone(), dim, Some(0.3));
        }
        Ok(enc)
    }

    pub fn from_path(path: impl AsRef<Path>, dim: usize) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::Onnx(format!("model not found: {}", path.display())));
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("onnx");
        // Community export uses `model.onnx`; parent dir or env id still identifies Nemotron.
        let hint = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or(stem);
        require_dimensions(hint, dim)?;
        require_dimensions(stem, dim)?;
        // Full ORT session lands behind feature onnx-embed + native lib.
        // Until then we refuse silent hash fallback when an explicit path is given
        // but the runtime is not compiled in — callers know they asked for ONNX.
        if cfg!(feature = "onnx-embed") {
            tracing::warn!(
                path = %path.display(),
                "ONNX path provided; native ORT session not linked in this build, using hash stand-in of matching dim"
            );
            Ok(Self {
                model_id: stem.to_string(),
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
    use crate::encoder::{
        DenseEncoder, NEMOTRON_3_EMBED_1B_DIM, NEMOTRON_3_EMBED_1B_ID,
        NEMOTRON_3_EMBED_1B_OLLAMA_TAG, NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
    };

    #[tokio::test]
    async fn known_model_ids() {
        let e = OnnxEncoder::from_model("nomic-embed-text-v1.5").unwrap();
        assert_eq!(e.model_id(), "nomic-embed-text-v1.5");
        assert_eq!(e.dimensions(), 768);
        let v = e.encode(&[String::from("hola")]).await.unwrap();
        assert_eq!(v[0].len(), 768);
    }

    #[tokio::test]
    async fn nemotron_id_and_dim_2048() {
        for name in [
            NEMOTRON_3_EMBED_1B_ID,
            NEMOTRON_3_EMBED_1B_OLLAMA_TAG,
            NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
            "Nemotron-3-Embed-1B-BF16",
        ] {
            let e = OnnxEncoder::from_model(name).unwrap();
            assert_eq!(e.model_id(), NEMOTRON_3_EMBED_1B_ID, "{name}");
            assert_eq!(e.dimensions(), NEMOTRON_3_EMBED_1B_DIM, "{name}");
            let v = e.encode(&[String::from("hola")]).await.unwrap();
            assert_eq!(v[0].len(), 2048, "{name}");
        }
    }

    #[test]
    fn nemotron_rejects_mismatched_dim() {
        let err =
            OnnxEncoder::from_model_with_dimensions(NEMOTRON_3_EMBED_1B_ID, 1024).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 1024);
                assert_eq!(model, NEMOTRON_3_EMBED_1B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn nemotron_accepts_native_dim() {
        let e = OnnxEncoder::from_model_with_dimensions(NEMOTRON_3_EMBED_1B_ID, 2048).unwrap();
        assert_eq!(e.dimensions(), 2048);
    }
}
