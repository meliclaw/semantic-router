//! ATIRE BM25 sparse encoder — port of semantic_router/encoders/bm25.py
//! Uses a hash tokenizer (no BERT download). Optional HF tokenizers behind feature.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::SparseEncoder;
use crate::error::{Error, Result};
use crate::route::Route;
use crate::schema::SparseEmbedding;

const DEFAULT_VOCAB: usize = 10_000;

#[derive(Debug, Clone)]
pub struct Bm25Encoder {
    name: String,
    k1: f32,
    b: f32,
    vocab_size: usize,
    corpus_size: Option<usize>,
    avg_doc_len: Option<f32>,
    /// df[token_id] = number of docs containing token
    df: Option<Vec<f32>>,
}

impl Default for Bm25Encoder {
    fn default() -> Self {
        Self::new("bm25", 1.5, 0.75, DEFAULT_VOCAB)
    }
}

impl Bm25Encoder {
    pub fn new(name: impl Into<String>, k1: f32, b: f32, vocab_size: usize) -> Self {
        Self {
            name: name.into(),
            k1,
            b,
            vocab_size,
            corpus_size: None,
            avg_doc_len: None,
            df: None,
        }
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|tok| {
                let mut hasher = DefaultHasher::new();
                tok.hash(&mut hasher);
                (hasher.finish() as usize % self.vocab_size) as u32
            })
            .collect()
    }

    fn tf(&self, docs: &[Vec<u32>]) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0f32; self.vocab_size]; docs.len()];
        for (i, ids) in docs.iter().enumerate() {
            for &id in ids {
                if id == 0 {
                    continue;
                }
                out[i][id as usize] += 1.0;
            }
            out[i][0] = 0.0;
        }
        out
    }

    pub fn fit(&mut self, routes: &[Route]) -> Result<()> {
        if routes.is_empty() {
            return Err(Error::msg("BM25 fit requires routes"));
        }
        let utterances: Vec<String> = routes
            .iter()
            .flat_map(|r| r.utterances.iter().cloned())
            .collect();
        let ids: Vec<Vec<u32>> = utterances.iter().map(|u| self.tokenize(u)).collect();
        let corpus = self.tf(&ids);
        self.corpus_size = Some(utterances.len());
        let doc_lengths: Vec<f32> = corpus.iter().map(|row| row.iter().sum()).collect();
        self.avg_doc_len = Some(doc_lengths.iter().sum::<f32>() / doc_lengths.len() as f32);
        let mut df = vec![0.0f32; self.vocab_size];
        for row in &corpus {
            for (j, &v) in row.iter().enumerate() {
                if v > 0.0 {
                    df[j] += 1.0;
                }
            }
        }
        df[0] = 0.0;
        self.df = Some(df);
        Ok(())
    }

    fn encode_queries_sync(&self, queries: &[String]) -> Result<Vec<SparseEmbedding>> {
        let df = self.df.as_ref().ok_or(Error::EncoderNotFitted)?;
        let n = self.corpus_size.ok_or(Error::EncoderNotFitted)? as f32;
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            let ids = self.tokenize(q);
            let mut idf = vec![0.0f32; self.vocab_size];
            let mut seen = vec![false; self.vocab_size];
            for &id in &ids {
                let i = id as usize;
                if i == 0 || seen[i] {
                    continue;
                }
                seen[i] = true;
                let mut dfi = df[i];
                if dfi > 0.0 {
                    dfi += 0.5;
                }
                if dfi != 0.0 {
                    idf[i] = ((n + 1.0) / dfi).ln();
                }
            }
            let sum: f32 = idf.iter().sum();
            if sum > 0.0 {
                for v in &mut idf {
                    *v /= sum;
                }
            }
            out.push(SparseEmbedding::from_dense(&idf));
        }
        Ok(out)
    }

    fn encode_documents_sync(&self, documents: &[String]) -> Result<Vec<SparseEmbedding>> {
        let avg = self.avg_doc_len.ok_or(Error::EncoderNotFitted)?;
        let ids: Vec<Vec<u32>> = documents.iter().map(|d| self.tokenize(d)).collect();
        let tf = self.tf(&ids);
        let mut out = Vec::with_capacity(tf.len());
        for row in tf {
            let tf_sum: f32 = row.iter().sum();
            let mut normed = vec![0.0f32; self.vocab_size];
            // ATIRE: tf / (tf + k1 * (1 - b + b * |D|/avgdl))
            let denom_base = self.k1 * (1.0 - self.b + self.b * (tf_sum / avg.max(1e-12)));
            for (j, &tfv) in row.iter().enumerate() {
                if tfv == 0.0 {
                    continue;
                }
                normed[j] = tfv / (denom_base + tfv);
            }
            out.push(SparseEmbedding::from_dense(&normed));
        }
        Ok(out)
    }
}

#[async_trait]
impl SparseEncoder for Bm25Encoder {
    fn name(&self) -> &str {
        &self.name
    }
    async fn encode_queries(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>> {
        self.encode_queries_sync(docs)
    }
    async fn encode_documents(&self, docs: &[String]) -> Result<Vec<SparseEmbedding>> {
        self.encode_documents_sync(docs)
    }
}
