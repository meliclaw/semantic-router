//! HybridRouter — dense × alpha + sparse × (1-alpha).
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use std::collections::HashMap;
use std::sync::Arc;

use super::{Aggregation, RouteRequest};
use crate::encoder::{DenseEncoder, SparseEncoder};
use crate::error::{Error, Result};
use crate::index::{HybridLocalIndex, Index};
use crate::route::Route;
use crate::schema::{RouteChoice, UtteranceRecord};

pub struct HybridRouter {
    encoder: Arc<dyn DenseEncoder>,
    sparse: Box<dyn SparseEncoder>,
    index: Box<dyn Index>,
    routes: Vec<Route>,
    top_k: usize,
    aggregation: Aggregation,
    score_threshold: Option<f32>,
    alpha: f32,
}

impl HybridRouter {
    pub fn new(
        encoder: Arc<dyn DenseEncoder>,
        sparse: Box<dyn SparseEncoder>,
        routes: Vec<Route>,
        alpha: f32,
    ) -> Self {
        let score_threshold = encoder.score_threshold().map(|t| t * alpha);
        Self {
            encoder,
            sparse,
            index: Box::new(HybridLocalIndex::new()),
            routes,
            top_k: 5,
            aggregation: Aggregation::Mean,
            score_threshold,
            alpha,
        }
    }

    pub async fn build(mut self) -> Result<Self> {
        let routes = std::mem::take(&mut self.routes);
        self.add(routes).await?;
        Ok(self)
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self.score_threshold = self.encoder.score_threshold().map(|t| t * alpha);
        self
    }

    pub async fn add(&mut self, routes: Vec<Route>) -> Result<()> {
        let mut texts = Vec::new();
        let mut meta: Vec<(String, String)> = Vec::new();
        for r in &routes {
            for u in &r.utterances {
                texts.push(u.clone());
                meta.push((r.name.clone(), u.clone()));
            }
        }
        let dense = self.encoder.encode_documents(&texts).await?;
        let sparse = self.sparse.encode_documents(&texts).await?;
        let records: Vec<UtteranceRecord> = meta
            .into_iter()
            .zip(dense)
            .zip(sparse)
            .map(|(((route, utterance), embedding), sp)| UtteranceRecord {
                route,
                utterance,
                embedding: scale(&embedding, self.alpha),
                sparse: Some(scale_sparse(sp, 1.0 - self.alpha)),
                metadata: Default::default(),
                function_schemas: None,
            })
            .collect();
        self.index.add(records).await?;
        self.routes.extend(routes);
        Ok(())
    }

    pub async fn route(&self, text: &str) -> Result<RouteChoice> {
        let mut dens = self.encoder.encode_queries(&[text.to_string()]).await?;
        let mut sps = self.sparse.encode_queries(&[text.to_string()]).await?;
        let mut d = dens.pop().ok_or_else(|| Error::msg("empty dense"))?;
        d = scale(&d, self.alpha);
        let mut s = sps.pop().ok_or_else(|| Error::msg("empty sparse"))?;
        s = scale_sparse(s, 1.0 - self.alpha);
        let (scores, routes) = self.index.query(&d, self.top_k, None, Some(&s)).await?;
        let scored = score_routes(&routes, &scores, self.aggregation);
        pass_routes(&self.routes, &scored, self.score_threshold, Some(1))
            .map(|v| v.into_iter().next().unwrap_or_default())
    }

    pub async fn route_with(&self, req: RouteRequest) -> Result<Vec<RouteChoice>> {
        let text = req.text.ok_or(Error::MissingQuery)?;
        let choice = self.route(&text).await?;
        Ok(vec![choice])
    }
}

fn scale(v: &[f32], a: f32) -> Vec<f32> {
    v.iter().map(|x| x * a).collect()
}

fn scale_sparse(mut s: crate::schema::SparseEmbedding, a: f32) -> crate::schema::SparseEmbedding {
    for v in &mut s.values {
        *v *= a;
    }
    s
}

fn score_routes(
    routes: &[String],
    scores: &[f32],
    aggregation: Aggregation,
) -> Vec<(String, f32, Vec<f32>)> {
    let mut by: HashMap<String, Vec<f32>> = HashMap::new();
    for (r, s) in routes.iter().zip(scores.iter()) {
        by.entry(r.clone()).or_default().push(*s);
    }
    let mut total: Vec<_> = by
        .into_iter()
        .map(|(route, sc)| {
            let agg = aggregation.apply(&sc);
            (route, agg, sc)
        })
        .collect();
    total.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    total
}

fn pass_routes(
    routes: &[Route],
    scored: &[(String, f32, Vec<f32>)],
    global: Option<f32>,
    limit: Option<usize>,
) -> Result<Vec<RouteChoice>> {
    let mut passed = Vec::new();
    for (name, total, _) in scored {
        let Some(route) = routes.iter().find(|r| &r.name == name) else {
            continue;
        };
        let threshold = route.score_threshold.or(global);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{Bm25Encoder, HashDenseEncoder};
    use crate::route::Route;

    #[tokio::test]
    async fn hybrid_routes_overlapping_terms() {
        let mut bm25 = Bm25Encoder::default();
        let routes = vec![
            Route::new("weather", vec!["what's the weather in london"]),
            Route::new("chitchat", vec!["how are you today"]),
        ];
        bm25.fit(&routes).unwrap();
        let mut router = HybridRouter::new(
            Arc::new(HashDenseEncoder::new("hash", 128, Some(0.01))),
            Box::new(bm25),
            routes,
            0.3,
        );
        router = router.build().await.unwrap();
        let choice = router.route("weather in london").await.unwrap();
        assert_eq!(choice.name.as_deref(), Some("weather"));
    }
}
