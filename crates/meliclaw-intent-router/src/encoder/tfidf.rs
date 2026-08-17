//! TF-IDF sparse encoder — port of semantic_router/encoders/tfidf.py
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use async_trait::async_trait;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::SparseEncoder;
use crate::error::{Error, Result};
use crate::route::Route;
use crate::schema::SparseEmbedding;

#[derive(Debug, Clone, Default)]
pub struct TfidfEncoder {
    name: String,
    word_index: BTreeMap<String, usize>,
    idf: Vec<f32>,
}

impl TfidfEncoder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            word_index: BTreeMap::new(),
            idf: Vec::new(),
        }
    }

    fn preprocess(doc: &str) -> String {
        doc.to_lowercase()
            .chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect()
    }

    pub fn fit(&mut self, routes: &[Route]) -> Result<()> {
        let docs: Vec<String> = routes
            .iter()
            .flat_map(|r| r.utterances.iter().map(|u| Self::preprocess(u)))
            .collect();
        let mut words = HashSet::new();
        for doc in &docs {
            for w in doc.split_whitespace() {
                words.insert(w.to_string());
            }
        }
        if words.is_empty() {
            return Err(Error::msg("Too little data to fit TfidfEncoder"));
        }
        self.word_index = words.into_iter().enumerate().map(|(i, w)| (w, i)).collect();
        let n = docs.len() as f32;
        let mut df = vec![0.0f32; self.word_index.len()];
        for doc in &docs {
            let uniq: HashSet<&str> = doc.split_whitespace().collect();
            for w in uniq {
                if let Some(&i) = self.word_index.get(w) {
                    df[i] += 1.0;
                }
            }
        }
        self.idf = df.iter().map(|d| (n / (d + 1.0)).ln()).collect();
        Ok(())
    }

    fn encode_sync(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>> {
        if self.word_index.is_empty() || self.idf.is_empty() {
            return Err(Error::EncoderNotFitted);
        }
        if docs.is_empty() {
            return Err(Error::msg("No documents to encode"));
        }
        let vsz = self.word_index.len();
        let mut out = Vec::with_capacity(docs.len());
        for doc in docs {
            let pre = Self::preprocess(doc);
            let mut tf = vec![0.0f32; vsz];
            let mut counts: HashMap<&str, f32> = HashMap::new();
            for w in pre.split_whitespace() {
                *counts.entry(w).or_insert(0.0) += 1.0;
            }
            for (w, c) in counts {
                if let Some(&i) = self.word_index.get(w) {
                    tf[i] = c;
                }
            }
            let norm = tf.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut tf {
                *x /= norm;
            }
            let tfidf: Vec<f32> = tf.iter().zip(self.idf.iter()).map(|(t, i)| t * i).collect();
            out.push(SparseEmbedding::from_dense(&tfidf));
        }
        Ok(out)
    }
}

#[async_trait]
impl SparseEncoder for TfidfEncoder {
    fn name(&self) -> &str {
        &self.name
    }
    async fn encode_queries(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>> {
        self.encode_sync(docs)
    }
    async fn encode_documents(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>> {
        self.encode_sync(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::Route;

    #[tokio::test]
    async fn tfidf_fits_and_encodes() {
        let mut enc = TfidfEncoder::new("tfidf");
        let routes = vec![
            Route::new("a", vec!["alpha document about cats"]),
            Route::new("b", vec!["beta document about dogs"]),
        ];
        enc.fit(&routes).unwrap();
        // IDF is ln(n/(df+1)); a term in every doc still has non-zero weight.
        let q = enc
            .encode_queries(&[String::from("document about cats")])
            .await
            .unwrap();
        assert_eq!(q.len(), 1);
        assert!(!q[0].indices.is_empty());
    }
}
