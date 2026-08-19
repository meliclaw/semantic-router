//! HTTP facade for `meliclaw-intent-router`.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post, put};
use axum::{Json, Router};
use meliclaw_intent_router::encoder::HashDenseEncoder;
use meliclaw_intent_router::memory_routes::memory_intent_routes;
use meliclaw_intent_router::route::Route;
use meliclaw_intent_router::schema::RouteChoice;
use meliclaw_intent_router::{OnnxEncoder, SemanticRouter};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

struct AppState {
    router: RwLock<SemanticRouter>,
}

#[derive(Deserialize)]
struct RouteBody {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    crate_name: &'static str,
    version: &'static str,
}

#[derive(Deserialize)]
struct PutRoutes {
    routes: Vec<Route>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "meliclaw_intent_router_service=info,tower_http=info".into()),
        )
        .init();

    let encoder_kind = std::env::var("MELICLAW_ENCODER").unwrap_or_else(|_| "hash".into());
    let mut builder = SemanticRouter::builder().routes(memory_intent_routes());
    builder = match encoder_kind.as_str() {
        "onnx" => {
            let model = std::env::var("MELICLAW_ONNX_MODEL_ID")
                .unwrap_or_else(|_| "nomic-embed-text-v1.5".into());
            builder.encoder(OnnxEncoder::from_model(&model).expect("onnx model id"))
        }
        _ => builder.encoder(HashDenseEncoder::default()),
    };
    let semantic = builder.build().await.expect("router init");

    let state = Arc::new(AppState {
        router: RwLock::new(semantic),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/route", post(route_handler))
        .route("/v1/routes", put(put_routes).get(get_routes))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = std::env::var("MELICLAW_INTENT_BIND").unwrap_or_else(|_| "0.0.0.0:8091".into());
    tracing::info!("meliclaw-intent-router-service listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        crate_name: "meliclaw-intent-router",
        version: meliclaw_intent_router::VERSION,
    })
}

async fn route_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RouteBody>,
) -> Result<Json<RouteChoice>, (StatusCode, String)> {
    let router = state.router.read().await;
    let mut choice = router
        .route(&body.query)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if body.limit == Some(0) {
        choice = RouteChoice::empty();
    }
    Ok(Json(choice))
}

async fn put_routes(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PutRoutes>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut router = state.router.write().await;
    router
        .add(body.routes)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn get_routes(State(state): State<Arc<AppState>>) -> Json<Vec<Route>> {
    let router = state.router.read().await;
    Json(router.routes().to_vec())
}
