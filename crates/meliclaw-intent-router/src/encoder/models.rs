//! Known dense embedding models for the Meliclaw Intent Classifier (Capa 1).
//!
//! Model **weights** are not MIT. NVIDIA Nemotron weights are OpenMDW-1.1;
//! see `THIRD_PARTY_NOTICES.md`. ColNomic Embed Multimodal 7B is not wired
//! here (late-interaction / visual retrieval, not a single dense vector).
//! Modified by Meliclaw, 2026.

use crate::error::{Error, Result};

/// Official Hugging Face / NVIDIA id for Nemotron-3-Embed-1B (BF16).
pub const NEMOTRON_3_EMBED_1B_ID: &str = "nvidia/Nemotron-3-Embed-1B-BF16";
/// NVIDIA NIM short name; also the Ollama / HTTP tag to serve under.
pub const NEMOTRON_3_EMBED_1B_OLLAMA_TAG: &str = "nemotron-3-embed-1b";
/// Community ONNX export (not NVIDIA-hosted). Verify checksum vs NVIDIA weights.
pub const NEMOTRON_3_EMBED_1B_ONNX_EXPORT: &str = "kzzalews/Nemotron-3-Embed-1B-BF16-onnx";
/// Default graph file in the community ONNX export (float32).
pub const NEMOTRON_3_EMBED_1B_ONNX_FILE: &str = "model.onnx";
/// FP16 graph in the community ONNX export (`fp16/model.onnx` + `.data`).
pub const NEMOTRON_3_EMBED_1B_ONNX_FP16_FILE: &str = "fp16/model.onnx";
/// Native embedding width. Matryoshka slices (1024/512) are not accepted here.
pub const NEMOTRON_3_EMBED_1B_DIM: usize = 2048;

const DEFAULT_THRESHOLD: f32 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseModelSpec {
    pub id: &'static str,
    pub dimensions: usize,
    pub score_threshold: f32,
}

/// Resolve a caller-supplied model name or alias to the catalog entry.
///
/// Unknown names return `None` so encoders keep their historical defaults.
pub fn resolve_dense_model(name: &str) -> Option<DenseModelSpec> {
    let spec = match name {
        "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => DenseModelSpec {
            id: "nomic-embed-text-v1.5",
            dimensions: 768,
            score_threshold: DEFAULT_THRESHOLD,
        },
        "bge-m3" | "BAAI/bge-m3" => DenseModelSpec {
            id: "bge-m3",
            dimensions: 1024,
            score_threshold: DEFAULT_THRESHOLD,
        },
        "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => DenseModelSpec {
            id: "bge-small-en-v1.5",
            dimensions: 384,
            score_threshold: DEFAULT_THRESHOLD,
        },
        "nvidia/Nemotron-3-Embed-1B-BF16"
        | "Nemotron-3-Embed-1B-BF16"
        | "nemotron-3-embed-1b"
        | "nemotron-3-embed-1b-bf16"
        | "nvidia/nemotron-3-embed-1b"
        | "kzzalews/Nemotron-3-Embed-1B-BF16-onnx" => DenseModelSpec {
            id: NEMOTRON_3_EMBED_1B_ID,
            dimensions: NEMOTRON_3_EMBED_1B_DIM,
            score_threshold: DEFAULT_THRESHOLD,
        },
        _ => return None,
    };
    Some(spec)
}

/// Error if `name` is a known model and `dim` is not its native width.
pub fn require_dimensions(name: &str, dim: usize) -> Result<()> {
    if let Some(spec) = resolve_dense_model(name) {
        if spec.dimensions != dim {
            return Err(Error::DimensionMismatch {
                model: spec.id.to_string(),
                expected: spec.dimensions,
                got: dim,
            });
        }
    }
    Ok(())
}

/// Error if a produced embedding vector does not match the encoder width.
pub fn expect_embedding_dim(model: &str, expected: usize, vector: &[f32]) -> Result<()> {
    if vector.len() != expected {
        return Err(Error::DimensionMismatch {
            model: model.to_string(),
            expected,
            got: vector.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nemotron_aliases_are_2048() {
        for name in [
            NEMOTRON_3_EMBED_1B_ID,
            "Nemotron-3-Embed-1B-BF16",
            NEMOTRON_3_EMBED_1B_OLLAMA_TAG,
            "nemotron-3-embed-1b-bf16",
            "nvidia/nemotron-3-embed-1b",
            NEMOTRON_3_EMBED_1B_ONNX_EXPORT,
        ] {
            let spec = resolve_dense_model(name).expect(name);
            assert_eq!(spec.id, NEMOTRON_3_EMBED_1B_ID, "{name}");
            assert_eq!(spec.dimensions, 2048, "{name}");
        }
    }

    #[test]
    fn nemotron_rejects_mismatched_dim() {
        let err = require_dimensions(NEMOTRON_3_EMBED_1B_ID, 768).unwrap_err();
        match err {
            Error::DimensionMismatch {
                expected,
                got,
                model,
            } => {
                assert_eq!(expected, 2048);
                assert_eq!(got, 768);
                assert_eq!(model, NEMOTRON_3_EMBED_1B_ID);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn nemotron_accepts_native_dim() {
        require_dimensions(NEMOTRON_3_EMBED_1B_OLLAMA_TAG, 2048).unwrap();
        expect_embedding_dim(NEMOTRON_3_EMBED_1B_ID, 2048, &vec![0.0; 2048]).unwrap();
    }

    #[test]
    fn unknown_model_allows_any_dim() {
        require_dimensions("custom-local-embed", 512).unwrap();
        assert!(resolve_dense_model("custom-local-embed").is_none());
    }
}
