//! LocalAI model allowlist and catalog-id → YAML `name:` mapping for the Capa 1 probe.
//!
//! jina-embeddings-v3 and NV-Embed-v2 are in the library catalog but **not** in
//! this CLI (NC weights; omitted by product). Qwen3-Embedding-0.6B is also omitted.

use meliclaw_intent_router::encoder::{
    resolve_dense_model, GTE_QWEN2_7B_INSTRUCT_ID, NEMOTRON_3_EMBED_1B_ID, QWEN3_EMBEDDING_8B_ID,
    QWEN3_VL_EMBEDDING_2B_ID, QWEN3_VL_EMBEDDING_8B_ID,
};

use crate::{CliError, EmbedMode};

/// Catalog ids allowed in `--mode text`.
pub const TEXT_CATALOG_IDS: &[&str] = &[
    "nomic-embed-text-v1.5",
    "bge-m3",
    "bge-small-en-v1.5",
    NEMOTRON_3_EMBED_1B_ID,
    QWEN3_VL_EMBEDDING_2B_ID,
    QWEN3_EMBEDDING_8B_ID,
    GTE_QWEN2_7B_INSTRUCT_ID,
];

/// Catalog ids allowed in `--mode multimodal`.
pub const MULTIMODAL_CATALOG_IDS: &[&str] = &[QWEN3_VL_EMBEDDING_2B_ID, QWEN3_VL_EMBEDDING_8B_ID];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub catalog_id: &'static str,
    pub dimensions: usize,
    /// LocalAI YAML `name:` / request `model` field (short alias).
    pub localai_name: &'static str,
}

/// Map a catalog id to the LocalAI YAML `name:` this CLI sends.
///
/// `--model` still accepts catalog ids and library aliases via `resolve_dense_model`.
pub fn localai_name_for_catalog_id(catalog_id: &str) -> Option<&'static str> {
    match catalog_id {
        "nomic-embed-text-v1.5" => Some("nomic-embed-text-v1.5"),
        "bge-m3" => Some("bge-m3"),
        "bge-small-en-v1.5" => Some("bge-small-en-v1.5"),
        id if id == NEMOTRON_3_EMBED_1B_ID => Some("nemotron-3-embed-1b"),
        id if id == QWEN3_VL_EMBEDDING_2B_ID => Some("qwen3-vl-embedding-2b"),
        id if id == QWEN3_EMBEDDING_8B_ID => Some("qwen3-embedding-8b"),
        id if id == GTE_QWEN2_7B_INSTRUCT_ID => Some("gte-qwen2-7b-instruct"),
        id if id == QWEN3_VL_EMBEDDING_8B_ID => Some("qwen3-vl-embedding-8b"),
        _ => None,
    }
}

pub fn allowed_ids(mode: EmbedMode) -> &'static [&'static str] {
    match mode {
        EmbedMode::Text => TEXT_CATALOG_IDS,
        EmbedMode::Multimodal => MULTIMODAL_CATALOG_IDS,
    }
}

pub fn resolve_cli_model(raw: &str, mode: EmbedMode) -> Result<ResolvedModel, CliError> {
    let spec = resolve_dense_model(raw).ok_or_else(|| {
        CliError::usage(format!(
            "unknown model `{raw}`. {} mode allows: {}",
            mode.as_str(),
            allowed_ids(mode).join(", ")
        ))
    })?;

    let allowed = allowed_ids(mode);
    if !allowed.contains(&spec.id) {
        return Err(mode_mismatch_error(raw, spec.id, mode));
    }

    let localai_name = localai_name_for_catalog_id(spec.id).ok_or_else(|| {
        CliError::usage(format!(
            "model `{}` has no LocalAI name mapping in this CLI",
            spec.id
        ))
    })?;

    Ok(ResolvedModel {
        catalog_id: spec.id,
        dimensions: spec.dimensions,
        localai_name,
    })
}

fn mode_mismatch_error(raw: &str, catalog_id: &str, mode: EmbedMode) -> CliError {
    let other = match mode {
        EmbedMode::Text => EmbedMode::Multimodal,
        EmbedMode::Multimodal => EmbedMode::Text,
    };
    let in_other = allowed_ids(other).contains(&catalog_id);
    let omitted = matches!(
        catalog_id,
        "jinaai/jina-embeddings-v3" | "nvidia/NV-Embed-v2" | "Qwen/Qwen3-Embedding-0.6B"
    );

    let mut msg = if omitted {
        format!(
            "model `{raw}` ({catalog_id}) is not in this CLI allowlist \
             (jina-v3 and NV-Embed-v2 are omitted; Qwen3-Embedding-0.6B is not a Capa 1 LocalAI probe target)"
        )
    } else if in_other {
        format!(
            "model `{raw}` ({catalog_id}) is not allowed in {} mode; use --mode {}",
            mode.as_str(),
            other.as_str()
        )
    } else {
        format!(
            "model `{raw}` ({catalog_id}) is not allowed in {} mode",
            mode.as_str()
        )
    };
    msg.push_str(&format!(
        ". {} mode allows: {}",
        mode.as_str(),
        allowed_ids(mode).join(", ")
    ));
    CliError::usage(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmbedMode;
    use meliclaw_intent_router::encoder::{
        JINA_EMBEDDINGS_V3_ID, NV_EMBED_V2_ID, QWEN3_EMBEDDING_0_6B_ID,
    };

    fn ok_id(raw: &str, mode: EmbedMode) -> &'static str {
        resolve_cli_model(raw, mode).unwrap().catalog_id
    }

    fn err_msg(raw: &str, mode: EmbedMode) -> String {
        resolve_cli_model(raw, mode).unwrap_err().to_string()
    }

    #[test]
    fn text_mode_allowlist() {
        for raw in [
            "nomic-embed-text-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
            "bge-m3",
            "BAAI/bge-m3",
            "bge-small-en-v1.5",
            NEMOTRON_3_EMBED_1B_ID,
            "nemotron-3-embed-1b",
            QWEN3_VL_EMBEDDING_2B_ID,
            "qwen3-vl-embedding-2b",
            QWEN3_EMBEDDING_8B_ID,
            "qwen3-embedding-8b",
            "qwen3-embedding:8b",
            GTE_QWEN2_7B_INSTRUCT_ID,
            "gte-qwen2-7b-instruct",
            "gte-Qwen2-7B-instruct",
        ] {
            let m = resolve_cli_model(raw, EmbedMode::Text).expect(raw);
            assert!(
                TEXT_CATALOG_IDS.contains(&m.catalog_id),
                "{raw} -> {}",
                m.catalog_id
            );
        }
    }

    #[test]
    fn multimodal_mode_allowlist() {
        let a = resolve_cli_model(QWEN3_VL_EMBEDDING_2B_ID, EmbedMode::Multimodal).unwrap();
        assert_eq!(a.catalog_id, QWEN3_VL_EMBEDDING_2B_ID);
        assert_eq!(a.dimensions, 2048);
        assert_eq!(a.localai_name, "qwen3-vl-embedding-2b");
        let b = resolve_cli_model("qwen3-vl-embedding-8b", EmbedMode::Multimodal).unwrap();
        assert_eq!(b.catalog_id, QWEN3_VL_EMBEDDING_8B_ID);
        assert_eq!(b.dimensions, 4096);
        assert_eq!(b.localai_name, "qwen3-vl-embedding-8b");
    }

    #[test]
    fn vl_8b_rejected_in_text_mode() {
        let msg = err_msg(QWEN3_VL_EMBEDDING_8B_ID, EmbedMode::Text);
        assert!(msg.contains("multimodal"), "{msg}");
        assert!(msg.contains("not allowed in text"), "{msg}");
    }

    #[test]
    fn embedding_8b_rejected_in_multimodal() {
        let msg = err_msg(QWEN3_EMBEDDING_8B_ID, EmbedMode::Multimodal);
        assert!(msg.contains("text"), "{msg}");
        assert!(msg.contains("not allowed in multimodal"), "{msg}");
    }

    #[test]
    fn gte_qwen2_rejected_in_multimodal() {
        let msg = err_msg(GTE_QWEN2_7B_INSTRUCT_ID, EmbedMode::Multimodal);
        assert!(msg.contains("text"), "{msg}");
        assert!(msg.contains("not allowed in multimodal"), "{msg}");
        let m = resolve_cli_model(GTE_QWEN2_7B_INSTRUCT_ID, EmbedMode::Text).unwrap();
        assert_eq!(m.catalog_id, GTE_QWEN2_7B_INSTRUCT_ID);
        assert_eq!(m.dimensions, 3584);
        assert_eq!(m.localai_name, "gte-qwen2-7b-instruct");
    }

    #[test]
    fn text_only_models_rejected_in_multimodal() {
        for raw in [
            "nomic-embed-text-v1.5",
            "bge-m3",
            "bge-small-en-v1.5",
            NEMOTRON_3_EMBED_1B_ID,
            GTE_QWEN2_7B_INSTRUCT_ID,
        ] {
            let msg = err_msg(raw, EmbedMode::Multimodal);
            assert!(msg.contains("not allowed in multimodal"), "{raw}: {msg}");
            assert!(msg.contains("--mode text"), "{raw}: {msg}");
        }
    }

    #[test]
    fn vl_2b_allowed_in_both_modes() {
        assert_eq!(
            ok_id("qwen3-vl-embedding-2b", EmbedMode::Text),
            QWEN3_VL_EMBEDDING_2B_ID
        );
        assert_eq!(
            ok_id(QWEN3_VL_EMBEDDING_2B_ID, EmbedMode::Multimodal),
            QWEN3_VL_EMBEDDING_2B_ID
        );
    }

    #[test]
    fn jina_and_nv_embed_omitted() {
        for (raw, mode) in [
            (JINA_EMBEDDINGS_V3_ID, EmbedMode::Text),
            ("jina-embeddings-v3", EmbedMode::Text),
            (NV_EMBED_V2_ID, EmbedMode::Text),
            (NV_EMBED_V2_ID, EmbedMode::Multimodal),
        ] {
            let msg = err_msg(raw, mode);
            assert!(
                msg.contains("not in this CLI allowlist") || msg.contains("omitted"),
                "{raw}: {msg}"
            );
        }
    }

    #[test]
    fn qwen3_0_6b_omitted() {
        let msg = err_msg(QWEN3_EMBEDDING_0_6B_ID, EmbedMode::Text);
        assert!(msg.contains("0.6B") || msg.contains("allowlist"), "{msg}");
    }

    #[test]
    fn unknown_model_lists_allowlist() {
        let msg = err_msg("totally-unknown-embed", EmbedMode::Text);
        assert!(msg.contains("unknown model"), "{msg}");
        assert!(msg.contains("nomic-embed-text-v1.5"), "{msg}");
    }

    #[test]
    fn localai_short_names() {
        assert_eq!(
            localai_name_for_catalog_id(NEMOTRON_3_EMBED_1B_ID),
            Some("nemotron-3-embed-1b")
        );
        assert_eq!(
            localai_name_for_catalog_id(QWEN3_EMBEDDING_8B_ID),
            Some("qwen3-embedding-8b")
        );
        assert_eq!(
            localai_name_for_catalog_id(GTE_QWEN2_7B_INSTRUCT_ID),
            Some("gte-qwen2-7b-instruct")
        );
        assert_eq!(
            localai_name_for_catalog_id("nomic-embed-text-v1.5"),
            Some("nomic-embed-text-v1.5")
        );
        let m = resolve_cli_model(NEMOTRON_3_EMBED_1B_ID, EmbedMode::Text).unwrap();
        assert_eq!(m.localai_name, "nemotron-3-embed-1b");
        assert_eq!(m.dimensions, 2048);
    }

    #[test]
    fn matrix_text_vs_multimodal() {
        let rows: &[(&str, bool, bool)] = &[
            ("nomic-embed-text-v1.5", true, false),
            ("bge-m3", true, false),
            ("bge-small-en-v1.5", true, false),
            (NEMOTRON_3_EMBED_1B_ID, true, false),
            (QWEN3_EMBEDDING_8B_ID, true, false),
            (GTE_QWEN2_7B_INSTRUCT_ID, true, false),
            (QWEN3_VL_EMBEDDING_2B_ID, true, true),
            (QWEN3_VL_EMBEDDING_8B_ID, false, true),
        ];
        for (id, text, multi) in rows {
            assert_eq!(
                resolve_cli_model(id, EmbedMode::Text).is_ok(),
                *text,
                "text {id}"
            );
            assert_eq!(
                resolve_cli_model(id, EmbedMode::Multimodal).is_ok(),
                *multi,
                "multimodal {id}"
            );
        }
    }
}
