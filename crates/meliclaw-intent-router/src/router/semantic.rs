//! SemanticRouter — encode → local/remote index → aggregate → threshold.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::sync::Arc;

use super::{Aggregation, RouteRequest};
use crate::config::RouterConfig;
use crate::encoder::DenseEncoder;
use crate::error::{Error, Result};
use crate::index::{Index, LocalIndex};
use crate::route::Route;
use crate::schema::{RouteChoice, SyncMode, UtteranceRecord};

pub struct SemanticRouter {
    encoder: Arc<dyn DenseEncoder>,
    index: Box<dyn Index>,
    routes: Vec<Route>,
    top_k: usize,
    aggregation: Aggregation,
    score_threshold: Option<f32>,
    #[allow(dead_code)]
    auto_sync: Option<SyncMode>,
}

pub struct SemanticRouterBuilder {
    encoder: Option<Arc<dyn DenseEncoder>>,
    index: Option<Box<dyn Index>>,
    routes: Vec<Route>,
    top_k: usize,
    aggregation: Aggregation,
    auto_sync: Option<SyncMode>,
}

impl SemanticRouter {
    pub fn builder() -> SemanticRouterBuilder {
        SemanticRouterBuilder {
            encoder: None,
            index: None,
            routes: Vec::new(),
            top_k: 5,
            aggregation: Aggregation::Mean,
            auto_sync: None,
        }
    }

    pub async fn from_config(config: RouterConfig, encoder: Arc<dyn DenseEncoder>) -> Result<Self> {
        Self::builder()
            .encoder_arc(encoder)
            .routes(config.routes)
            .build()
            .await
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    pub fn score_threshold(&self) -> Option<f32> {
        self.score_threshold
    }

    pub fn set_threshold(&mut self, threshold: f32, route_name: Option<&str>) {
        match route_name {
            None => {
                for r in &mut self.routes {
                    r.score_threshold = Some(threshold);
                }
                self.score_threshold = Some(threshold);
            }
            Some(name) => {
                if let Some(r) = self.routes.iter_mut().find(|r| r.name == name) {
                    r.score_threshold = Some(threshold);
                }
            }
        }
    }

    pub fn get_thresholds(&self) -> HashMap<String, f32> {
        self.routes
            .iter()
            .map(|r| {
                (
                    r.name.clone(),
                    r.score_threshold.or(self.score_threshold).unwrap_or(0.0),
                )
            })
            .collect()
    }

    pub fn to_config(&self) -> RouterConfig {
        RouterConfig {
            encoder_type: self.encoder.encoder_type().to_string(),
            encoder_name: Some(self.encoder.name().to_string()),
            routes: self.routes.clone(),
        }
    }

    pub async fn add(&mut self, routes: Vec<Route>) -> Result<()> {
        let records = self.encode_routes(&routes).await?;
        self.index.add(records).await?;
        self.routes.extend(routes);
        Ok(())
    }

    pub async fn delete(&mut self, route_name: &str) -> Result<()> {
        self.index.delete_route(route_name).await?;
        self.routes.retain(|r| r.name != route_name);
        Ok(())
    }

    async fn encode_routes(&self, routes: &[Route]) -> Result<Vec<UtteranceRecord>> {
        let mut texts = Vec::new();
        let mut meta: Vec<(String, String)> = Vec::new();
        for r in routes {
            for u in &r.utterances {
                texts.push(u.clone());
                meta.push((r.name.clone(), u.clone()));
            }
        }
        let embs = self.encoder.encode_documents(&texts).await?;
        Ok(meta
            .into_iter()
            .zip(embs)
            .map(|((route, utterance), embedding)| UtteranceRecord {
                route,
                utterance,
                embedding,
                sparse: None,
                metadata: Default::default(),
                function_schemas: None,
            })
            .collect())
    }

    pub async fn route(&self, text: &str) -> Result<RouteChoice> {
        let out = self
            .route_with(RouteRequest {
                text: Some(text.to_string()),
                limit: Some(1),
                ..Default::default()
            })
            .await?;
        Ok(out.into_iter().next().unwrap_or_default())
    }

    pub async fn route_with(&self, req: RouteRequest) -> Result<Vec<RouteChoice>> {
        if !self.index.is_ready() {
            return Err(Error::IndexNotReady);
        }
        let vector = if let Some(v) = req.vector {
            v
        } else {
            let text = req.text.as_ref().ok_or(Error::MissingQuery)?;
            let mut embs = self.encoder.encode_queries(&[text.clone()]).await?;
            embs.pop().ok_or_else(|| Error::msg("empty encoding"))?
        };
        let filter = req.route_filter.as_deref();
        let (scores, routes) = self.index.query(&vector, self.top_k, filter, None).await?;
        let scored = self.score_routes(&routes, &scores);
        self.pass_routes(&scored, req.simulate_static, req.limit)
    }

    fn score_routes(&self, routes: &[String], scores: &[f32]) -> Vec<(String, f32, Vec<f32>)> {
        let mut by_class: HashMap<String, Vec<f32>> = HashMap::new();
        for (r, s) in routes.iter().zip(scores.iter()) {
            by_class.entry(r.clone()).or_default().push(*s);
        }
        let mut total: Vec<(String, f32, Vec<f32>)> = by_class
            .into_iter()
            .map(|(route, sc)| {
                let agg = self.aggregation.apply(&sc);
                (route, agg, sc)
            })
            .collect();
        total.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        total
    }

    fn pass_routes(
        &self,
        scored: &[(String, f32, Vec<f32>)],
        _simulate_static: bool,
        limit: Option<usize>,
    ) -> Result<Vec<RouteChoice>> {
        let mut passed = Vec::new();
        for (name, total, _) in scored {
            let route = self.routes.iter().find(|r| &r.name == name);
            let Some(route) = route else {
                continue;
            };
            let threshold = route.score_threshold.or(self.score_threshold);
            let ok = match threshold {
                Some(t) => *total >= t,
                None => true,
            };
            if ok {
                passed.push(route.choose(*total));
            }
            if let Some(lim) = limit {
                if passed.len() >= lim {
                    break;
                }
            }
        }
        if passed.is_empty() {
            passed.push(RouteChoice::empty());
        }
        Ok(passed)
    }

    pub async fn evaluate(&self, x: &[String], y: &[&str]) -> Result<f32> {
        let mut correct = 0usize;
        for (q, target) in x.iter().zip(y.iter()) {
            let choice = self.route(q).await?;
            if choice.name.as_deref() == Some(*target) {
                correct += 1;
            }
        }
        Ok(correct as f32 / x.len().max(1) as f32)
    }

    /// Random search over per-route thresholds (Python `fit`, not gradient).
    pub async fn fit(&mut self, x: &[String], y: &[&str], max_iter: usize) -> Result<f32> {
        let mut best_acc = self.evaluate(x, y).await?;
        let mut best = self.get_thresholds();
        for _ in 0..max_iter {
            let candidate = threshold_random_search(&best, 0.8);
            for (name, t) in &candidate {
                self.set_threshold(*t, Some(name));
            }
            let acc = self.evaluate(x, y).await?;
            if acc > best_acc {
                best_acc = acc;
                best = candidate;
            }
        }
        for (name, t) in &best {
            self.set_threshold(*t, Some(name));
        }
        Ok(best_acc)
    }
}

impl SemanticRouterBuilder {
    pub fn encoder<E: DenseEncoder + 'static>(mut self, encoder: E) -> Self {
        self.encoder = Some(Arc::new(encoder));
        self
    }

    pub fn encoder_arc(mut self, encoder: Arc<dyn DenseEncoder>) -> Self {
        self.encoder = Some(encoder);
        self
    }

    pub fn index<I: Index + 'static>(mut self, index: I) -> Self {
        self.index = Some(Box::new(index));
        self
    }

    pub fn routes(mut self, routes: Vec<Route>) -> Self {
        self.routes = routes;
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn aggregation(mut self, aggregation: Aggregation) -> Self {
        self.aggregation = aggregation;
        self
    }

    pub fn auto_sync(mut self, mode: SyncMode) -> Self {
        self.auto_sync = Some(mode);
        self
    }

    pub async fn build(self) -> Result<SemanticRouter> {
        let encoder = self
            .encoder
            .ok_or_else(|| Error::msg("encoder is required"))?;
        let index = self.index.unwrap_or_else(|| Box::new(LocalIndex::new()));
        let mut routes = self.routes;
        let score_threshold = encoder.score_threshold();
        for r in &mut routes {
            if r.score_threshold.is_none() {
                r.score_threshold = score_threshold;
            }
        }
        let mut router = SemanticRouter {
            encoder,
            index,
            routes: Vec::new(),
            top_k: self.top_k,
            aggregation: self.aggregation,
            score_threshold,
            auto_sync: self.auto_sync,
        };
        if !routes.is_empty() {
            router.add(routes).await?;
        }
        let _ = router.index.is_ready();
        Ok(router)
    }
}

fn threshold_random_search(
    current: &HashMap<String, f32>,
    search_range: f32,
) -> HashMap<String, f32> {
    let mut rng = thread_rng();
    let mut out = HashMap::new();
    for (name, &thr) in current {
        let start = (thr - search_range).max(0.0);
        let stop = (thr + search_range).min(1.0);
        let mut grid = Vec::with_capacity(100);
        for i in 0..100 {
            let t = start + (stop - start) * (i as f32) / 99.0;
            grid.push(t);
        }
        let chosen = *grid.choose(&mut rng).unwrap_or(&thr);
        out.insert(name.clone(), chosen);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::HashDenseEncoder;
    use crate::memory_routes::memory_intent_routes;
    use crate::route::Route;

    #[tokio::test]
    async fn classifies_factual_nif() {
        let router = SemanticRouter::builder()
            .encoder(HashDenseEncoder::new("hash", 256, Some(0.05)))
            .routes(memory_intent_routes())
            .build()
            .await
            .unwrap();
        let choice = router
            .route("¿Cuál es el NIF del cliente Y?")
            .await
            .unwrap();
        assert_eq!(choice.name.as_deref(), Some("factual"));
        assert!(choice.similarity_score.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn fit_runs_and_returns_accuracy() {
        let mut router = SemanticRouter::builder()
            .encoder(HashDenseEncoder::new("hash", 128, Some(0.1)))
            .routes(vec![
                Route::new("a", vec!["alpha query example"]),
                Route::new("b", vec!["beta query example"]),
            ])
            .build()
            .await
            .unwrap();
        let x = vec!["alpha query example".into(), "beta query example".into()];
        let y = ["a", "b"];
        let acc = router.fit(&x, &y, 8).await.unwrap();
        assert!(acc >= 0.0 && acc <= 1.0);
        assert!(!router.get_thresholds().is_empty());
    }

    #[tokio::test]
    async fn empty_when_below_threshold() {
        let routes = vec![Route::new("x", vec!["hello world"]).with_threshold(0.99)];
        let router = SemanticRouter::builder()
            .encoder(HashDenseEncoder::new("hash", 64, Some(0.99)))
            .routes(routes)
            .build()
            .await
            .unwrap();
        let choice = router.route("zzzz totally unrelated").await.unwrap();
        assert!(choice.name.is_none());
    }
}
