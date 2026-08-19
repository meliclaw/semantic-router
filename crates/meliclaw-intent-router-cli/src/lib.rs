//! LocalAI embeddings probe for Meliclaw Capa 1.
//! Talks to LocalAI via [`meliclaw_intent_router::OpenAiEncoder`] (OpenAI-compatible
//! `/v1/embeddings`). Does not use `OllamaEncoder`.

mod allowlist;

use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};

use base64::Engine;
use clap::{Parser, ValueEnum};
use meliclaw_intent_router::encoder::{expect_embedding_dim, DenseEncoder};
use meliclaw_intent_router::OpenAiEncoder;
use serde::Serialize;

pub use allowlist::{
    localai_name_for_catalog_id, resolve_cli_model, ResolvedModel, MULTIMODAL_CATALOG_IDS,
    TEXT_CATALOG_IDS,
};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";
pub const DEFAULT_API_KEY: &str = "sk-local";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EmbedMode {
    Text,
    Multimodal,
}

impl EmbedMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Multimodal => "multimodal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Json,
    Table,
}

#[derive(Debug, Parser)]
#[command(
    name = "meliclaw-intent-embed",
    about = "Probe LocalAI embeddings for Meliclaw Capa 1 (OpenAI-compatible /v1/embeddings)"
)]
pub struct Cli {
    /// `text` (dense utterance models) or `multimodal` (Qwen3-VL-Embedding).
    #[arg(long, value_enum)]
    pub mode: EmbedMode,

    /// Catalog id or alias (e.g. `bge-m3`, `Qwen/Qwen3-Embedding-8B`).
    #[arg(long)]
    pub model: String,

    /// Text to embed. Combine with `--file`; otherwise stdin if it is not a TTY.
    #[arg(long)]
    pub text: Option<String>,

    /// UTF-8 file whose contents are the text to embed.
    #[arg(long)]
    pub file: Option<PathBuf>,

    /// Image file (repeatable). Multimodal only. Encoded as a data URL.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,

    /// LocalAI OpenAI base (`…/v1`). Request path is `{base}/embeddings`.
    #[arg(long, env = "LOCALAI_BASE_URL", default_value = DEFAULT_BASE_URL)]
    pub base_url: String,

    /// Dummy LocalAI key. `OPENAI_API_KEY` then `LOCALAI_API_KEY`, else `sk-local`.
    #[arg(long)]
    pub api_key: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    pub output: OutputFormat,

    /// Include the full embedding vector in JSON/table output.
    #[arg(long)]
    pub full_vector: bool,
}

#[derive(Debug)]
pub enum CliError {
    Usage(String),
    Io(io::Error),
    Router(meliclaw_intent_router::Error),
}

impl CliError {
    pub fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }

    pub fn exit_code(&self) -> i32 {
        1
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(m) => write!(f, "{m}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Router(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<io::Error> for CliError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<meliclaw_intent_router::Error> for CliError {
    fn from(e: meliclaw_intent_router::Error) -> Self {
        Self::Router(e)
    }
}

#[derive(Debug, Serialize)]
pub struct EmbedReport {
    pub model: String,
    pub localai_model: String,
    pub mode: String,
    pub dimensions: usize,
    pub vector_length: usize,
    pub l2_norm: f32,
    pub preview: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

pub fn resolve_api_key(flag: Option<&str>) -> String {
    if let Some(k) = flag {
        if !k.is_empty() {
            return k.to_string();
        }
    }
    for var in ["OPENAI_API_KEY", "LOCALAI_API_KEY"] {
        if let Ok(k) = std::env::var(var) {
            if !k.is_empty() {
                return k;
            }
        }
    }
    DEFAULT_API_KEY.into()
}

pub fn mime_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        _ => "image/png",
    }
}

pub fn image_data_url(mime: &str, bytes: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{b64}")
}

pub fn validate_inputs(
    mode: EmbedMode,
    text: Option<&str>,
    images: &[PathBuf],
) -> Result<(), CliError> {
    if mode == EmbedMode::Text && !images.is_empty() {
        return Err(CliError::usage(
            "--image is not allowed in --mode text; use --mode multimodal",
        ));
    }
    let has_text = text.map(|t| !t.trim().is_empty()).unwrap_or(false);
    if mode == EmbedMode::Multimodal && !has_text && images.is_empty() {
        return Err(CliError::usage(
            "multimodal mode requires --text/--file/stdin and/or at least one --image",
        ));
    }
    if mode == EmbedMode::Text && !has_text {
        return Err(CliError::usage(
            "text mode requires --text, --file, or stdin",
        ));
    }
    Ok(())
}

fn read_text(cli: &Cli) -> Result<Option<String>, CliError> {
    let mut parts = Vec::new();
    if let Some(t) = &cli.text {
        parts.push(t.clone());
    }
    if let Some(path) = &cli.file {
        parts.push(std::fs::read_to_string(path)?);
    }
    if parts.is_empty() {
        let stdin = io::stdin();
        if !stdin.is_terminal() {
            let mut buf = String::new();
            stdin.lock().read_to_string(&mut buf)?;
            if !buf.trim().is_empty() {
                parts.push(buf);
            }
        }
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parts.join("\n")))
    }
}

fn load_image_data_urls(paths: &[PathBuf]) -> Result<Vec<String>, CliError> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(path)?;
        out.push(image_data_url(mime_for_path(path), &bytes));
    }
    Ok(out)
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn format_report(report: &EmbedReport, format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".into()),
        OutputFormat::Table => {
            let preview = report
                .preview
                .iter()
                .map(|x| format!("{x:.6}"))
                .collect::<Vec<_>>()
                .join(", ");
            let mut lines = vec![
                format!("model            {}", report.model),
                format!("localai_model    {}", report.localai_model),
                format!("mode             {}", report.mode),
                format!("dimensions       {}", report.dimensions),
                format!("vector_length    {}", report.vector_length),
                format!("l2_norm          {:.6}", report.l2_norm),
                format!("preview          [{preview}]"),
            ];
            if let Some(v) = &report.vector {
                lines.push(format!(
                    "vector           [{} floats; use --output json --full-vector]",
                    v.len()
                ));
            }
            lines.join("\n")
        }
    }
}

pub fn build_report(
    resolved: &ResolvedModel,
    mode: EmbedMode,
    vector: Vec<f32>,
    full_vector: bool,
) -> EmbedReport {
    let preview: Vec<f32> = vector.iter().copied().take(8).collect();
    EmbedReport {
        model: resolved.catalog_id.to_string(),
        localai_model: resolved.localai_name.to_string(),
        mode: mode.as_str().to_string(),
        dimensions: resolved.dimensions,
        vector_length: vector.len(),
        l2_norm: l2_norm(&vector),
        preview,
        vector: full_vector.then_some(vector),
    }
}

pub async fn run(cli: Cli) -> Result<EmbedReport, CliError> {
    let resolved = resolve_cli_model(&cli.model, cli.mode)?;
    let text = read_text(&cli)?;
    validate_inputs(cli.mode, text.as_deref(), &cli.images)?;
    let images = if cli.mode == EmbedMode::Multimodal {
        load_image_data_urls(&cli.images)?
    } else {
        Vec::new()
    };

    let api_key = resolve_api_key(cli.api_key.as_deref());
    let encoder = OpenAiEncoder::new(resolved.localai_name, api_key)
        .with_base_url(&cli.base_url)
        .with_dimensions(resolved.dimensions)?;

    let texts: Vec<String> = text.into_iter().collect();
    let vectors = if images.is_empty() {
        encoder.encode(&texts).await?
    } else {
        encoder.encode_with_images(&texts, &images).await?
    };
    let vector = vectors
        .into_iter()
        .next()
        .ok_or_else(|| CliError::usage("LocalAI returned no embedding vectors".to_string()))?;
    expect_embedding_dim(resolved.catalog_id, resolved.dimensions, &vector)?;
    Ok(build_report(&resolved, cli.mode, vector, cli.full_vector))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use meliclaw_intent_router::OpenAiEncoder;
    use std::io::Write;

    #[test]
    fn clap_requires_mode_and_model() {
        let err = Cli::try_parse_from(["meliclaw-intent-embed"]).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("mode") || s.contains("required"), "{s}");
    }

    #[test]
    fn clap_parses_text_flags() {
        let c = Cli::try_parse_from([
            "meliclaw-intent-embed",
            "--mode",
            "text",
            "--model",
            "bge-m3",
            "--text",
            "hello",
        ])
        .unwrap();
        assert_eq!(c.mode, EmbedMode::Text);
        assert_eq!(c.model, "bge-m3");
        assert_eq!(c.text.as_deref(), Some("hello"));
        assert_eq!(c.base_url, DEFAULT_BASE_URL);
        assert_eq!(c.output, OutputFormat::Json);
        assert!(!c.full_vector);
    }

    #[test]
    fn clap_parses_repeatable_images() {
        let c = Cli::try_parse_from([
            "meliclaw-intent-embed",
            "--mode",
            "multimodal",
            "--model",
            "Qwen/Qwen3-VL-Embedding-2B",
            "--image",
            "a.png",
            "--image",
            "b.jpg",
        ])
        .unwrap();
        assert_eq!(c.images.len(), 2);
    }

    #[test]
    fn text_mode_rejects_images() {
        let err =
            validate_inputs(EmbedMode::Text, Some("hi"), &[PathBuf::from("x.png")]).unwrap_err();
        assert!(err.to_string().contains("--image"), "{err}");
    }

    #[test]
    fn multimodal_allows_text_only() {
        validate_inputs(EmbedMode::Multimodal, Some("hi"), &[]).unwrap();
    }

    #[test]
    fn multimodal_allows_image_only() {
        validate_inputs(EmbedMode::Multimodal, None, &[PathBuf::from("x.png")]).unwrap();
    }

    #[test]
    fn multimodal_rejects_empty() {
        let err = validate_inputs(EmbedMode::Multimodal, None, &[]).unwrap_err();
        assert!(err.to_string().contains("requires"), "{err}");
    }

    #[test]
    fn text_mode_rejects_empty() {
        let err = validate_inputs(EmbedMode::Text, None, &[]).unwrap_err();
        assert!(err.to_string().contains("text mode requires"), "{err}");
    }

    #[test]
    fn data_url_prefix() {
        let url = image_data_url("image/png", &[0x89, b'P', b'N', b'G']);
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        assert!(!url.contains("data:image/png;base64,data:"));
    }

    #[test]
    fn mime_from_extension() {
        assert_eq!(mime_for_path(Path::new("a.JPG")), "image/jpeg");
        assert_eq!(mime_for_path(Path::new("a.webp")), "image/webp");
        assert_eq!(mime_for_path(Path::new("a.bin")), "image/png");
    }

    #[test]
    fn report_preview_is_eight_and_hides_vector() {
        let resolved = ResolvedModel {
            catalog_id: "bge-m3",
            dimensions: 1024,
            localai_name: "bge-m3",
        };
        let v = vec![0.5; 1024];
        let r = build_report(&resolved, EmbedMode::Text, v, false);
        assert_eq!(r.preview.len(), 8);
        assert!(r.vector.is_none());
        assert_eq!(r.vector_length, 1024);
        assert!((r.l2_norm - 16.0).abs() < 1e-3); // sqrt(1024 * 0.25) = 16
        let json = format_report(&r, OutputFormat::Json);
        assert!(json.contains("\"preview\""));
        assert!(!json.contains("\"vector\""));
    }

    #[test]
    fn report_full_vector_included() {
        let resolved = ResolvedModel {
            catalog_id: "bge-m3",
            dimensions: 4,
            localai_name: "bge-m3",
        };
        let r = build_report(&resolved, EmbedMode::Text, vec![1.0, 0.0, 0.0, 0.0], true);
        assert_eq!(r.vector.as_ref().map(|v| v.len()), Some(4));
    }

    #[test]
    fn encoder_uses_catalog_dim_for_cli_models() {
        let e = OpenAiEncoder::new("nemotron-3-embed-1b", "sk-local")
            .with_base_url("http://127.0.0.1:8080/v1");
        assert_eq!(e.dimensions(), 2048);
        assert_eq!(e.name(), "nemotron-3-embed-1b");
        let e = OpenAiEncoder::new("qwen3-embedding-8b", "sk")
            .with_dimensions(4096)
            .unwrap();
        assert_eq!(e.dimensions(), 4096);
    }

    #[test]
    fn multimodal_request_body_keeps_text_encode_unchanged() {
        use meliclaw_intent_router::encoder::embeddings_request_body;
        let text = embeddings_request_body("bge-m3", &["hola".into()], &[]);
        assert!(text["input"].is_array());
        let multi = embeddings_request_body(
            "qwen3-vl-embedding-2b",
            &["cap".into()],
            &["data:image/png;base64,xx".into()],
        );
        assert!(multi["input"].is_object());
        assert_eq!(multi["input"]["images"][0], "data:image/png;base64,xx");
    }

    #[test]
    fn api_key_flag_wins() {
        assert_eq!(resolve_api_key(Some("sk-from-flag")), "sk-from-flag");
        assert_eq!(resolve_api_key(Some("")), DEFAULT_API_KEY);
    }

    #[tokio::test]
    async fn mock_http_dim_mismatch_exits_path() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
            let body = r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, resp.as_bytes()).await;
        });

        let encoder =
            OpenAiEncoder::new("bge-m3", "sk-local").with_base_url(format!("http://{addr}"));
        let err = encoder.encode(&["hi".into()]).await.unwrap_err();
        match err {
            meliclaw_intent_router::Error::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 1024);
                assert_eq!(got, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn mock_http_success_native_dim() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
            let vec = vec![0.0f32; 384];
            let body = serde_json::json!({ "data": [{ "embedding": vec }] }).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, resp.as_bytes()).await;
        });

        let encoder = OpenAiEncoder::new("bge-small-en-v1.5", "sk-local")
            .with_base_url(format!("http://{addr}"));
        let out = encoder.encode(&["hi".into()]).await.unwrap();
        assert_eq!(out[0].len(), 384);
    }

    #[tokio::test]
    async fn mock_http_error_status() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
            let body = "nope";
            let resp = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, resp.as_bytes()).await;
        });

        let encoder =
            OpenAiEncoder::new("bge-m3", "sk-local").with_base_url(format!("http://{addr}"));
        let err = encoder.encode(&["hi".into()]).await.unwrap_err();
        match err {
            meliclaw_intent_router::Error::Http(m) => assert!(m.contains("500"), "{m}"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn load_image_builds_data_url() {
        let dir = std::env::temp_dir();
        let path = dir.join("meliclaw-intent-embed-test.png");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0x89, b'P', b'N', b'G']).unwrap();
        }
        let urls = load_image_data_urls(std::slice::from_ref(&path)).unwrap();
        assert!(urls[0].starts_with("data:image/png;base64,"));
        let _ = std::fs::remove_file(path);
    }
}
