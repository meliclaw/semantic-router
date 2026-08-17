//! Qdrant REST index (on-prem, Apache-2.0 server). Feature `qdrant`.
//! No Pinecone. Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use async_trait::async_trait;
use serde_json::{json, Value};

use super::Index;
use crate::error::{Error, Result};
use crate::schema::{SparseEmbedding, Utterance, UtteranceRecord};

pub struct QdrantIndex {
    client: reqwest::Client,
    base: String,
    collection: String,
    dimensions: usize,
}

impl QdrantIndex {
    pub fn new(base: impl Into<String>, collection: impl Into<String>, dimensions: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base: base.into().trim_end_matches('/').to_string(),
            collection: collection.into(),
            dimensions,
        }
    }

    pub async fn ensure_collection(&self) -> Result<()> {
        let url = format!("{}/collections/{}", self.base, self.collection);
        let body = json!({
            "vectors": { "size": self.dimensions, "distance": "Cosine" }
        });
        let resp = self
            .client
            .put(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() && resp.status().as_u16() != 409 {
            return Err(Error::Http(format!(
                "qdrant create collection: {}",
                resp.status()
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl Index for QdrantIndex {
    fn index_type(&self) -> &'static str {
        "qdrant"
    }
    fn is_ready(&self) -> bool {
        true
    }
    fn len(&self) -> usize {
        0
    }

    async fn add(&mut self, records: Vec<UtteranceRecord>) -> Result<()> {
        self.ensure_collection().await?;
        let points: Vec<Value> = records
            .iter()
            .map(|r| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                use std::hash::{Hash, Hasher};
                r.route.hash(&mut hasher);
                r.utterance.hash(&mut hasher);
                json!({
                    "id": hasher.finish(),
                    "vector": r.embedding,
                    "payload": {
                        "sr_route": r.route,
                        "sr_utterance": r.utterance,
                    }
                })
            })
            .collect();
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.base, self.collection
        );
        let resp = self
            .client
            .put(&url)
            .json(&json!({ "points": points }))
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Http(format!("qdrant upsert: {}", resp.status())));
        }
        Ok(())
    }

    async fn query(
        &self,
        vector: &[f32],
        top_k: usize,
        route_filter: Option<&[String]>,
        _sparse: Option<&SparseEmbedding>,
    ) -> Result<(Vec<f32>, Vec<String>)> {
        let mut body = json!({
            "vector": vector,
            "limit": top_k,
            "with_payload": true,
        });
        if let Some(filter) = route_filter {
            body["filter"] = json!({
                "must": [{ "key": "sr_route", "match": { "any": filter } }]
            });
        }
        let url = format!(
            "{}/collections/{}/points/search",
            self.base, self.collection
        );
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Http(format!("qdrant search: {}", resp.status())));
        }
        let v: Value = resp.json().await.map_err(|e| Error::Http(e.to_string()))?;
        let mut scores = Vec::new();
        let mut routes = Vec::new();
        if let Some(arr) = v.get("result").and_then(|r| r.as_array()) {
            for hit in arr {
                if let Some(s) = hit.get("score").and_then(|s| s.as_f64()) {
                    scores.push(s as f32);
                }
                let route = hit
                    .pointer("/payload/sr_route")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                routes.push(route);
            }
        }
        Ok((scores, routes))
    }

    async fn delete_route(&mut self, route_name: &str) -> Result<()> {
        let url = format!(
            "{}/collections/{}/points/delete?wait=true",
            self.base, self.collection
        );
        let body = json!({
            "filter": { "must": [{ "key": "sr_route", "match": { "value": route_name } }] }
        });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Http(format!("qdrant delete: {}", resp.status())));
        }
        Ok(())
    }

    async fn get_utterances(&self, _include_metadata: bool) -> Result<Vec<Utterance>> {
        Ok(Vec::new())
    }

    async fn clear(&mut self) -> Result<()> {
        let url = format!("{}/collections/{}", self.base, self.collection);
        let _ = self.client.delete(&url).send().await;
        Ok(())
    }
}
