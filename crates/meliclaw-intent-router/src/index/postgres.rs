//! Postgres/pgvector index. Feature `postgres`.
//! Anclar imagen `pgvector/pgvector:pg17` — nunca `postgres:latest`.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use async_trait::async_trait;
use tokio_postgres::{Client, NoTls};

use super::Index;
use crate::error::{Error, Result};
use crate::schema::{SparseEmbedding, Utterance, UtteranceRecord};

pub struct PostgresIndex {
    client: Client,
    table: String,
    dimensions: usize,
    ready: bool,
}

fn validate_ident(name: &str) -> Result<&str> {
    let ok = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if ok {
        Ok(name)
    } else {
        Err(Error::Database(format!("invalid SQL identifier: {name}")))
    }
}

impl PostgresIndex {
    pub async fn connect(dsn: &str, table: &str, dimensions: usize) -> Result<Self> {
        let table = validate_ident(table)?.to_string();
        let (client, connection) = tokio_postgres::connect(dsn, NoTls)
            .await
            .map_err(|e| Error::Database(e.to_string()))?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let mut idx = Self {
            client,
            table,
            dimensions,
            ready: false,
        };
        idx.init_schema().await?;
        idx.ready = true;
        Ok(idx)
    }

    async fn init_schema(&self) -> Result<()> {
        let t = validate_ident(&self.table)?;
        let d = self.dimensions;
        if d == 0 || d > 16_000 {
            return Err(Error::Database("invalid embedding dimensions".into()));
        }
        let q = format!(
            "CREATE EXTENSION IF NOT EXISTS vector;
             CREATE TABLE IF NOT EXISTS {t} (
               id BIGSERIAL PRIMARY KEY,
               route TEXT NOT NULL,
               utterance TEXT NOT NULL,
               embedding vector({d}),
               metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
               function_schemas JSONB
             );
             CREATE INDEX IF NOT EXISTS {t}_route_idx ON {t}(route);"
        );
        self.client
            .batch_execute(&q)
            .await
            .map_err(|e| Error::Database(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl Index for PostgresIndex {
    fn index_type(&self) -> &'static str {
        "postgres"
    }
    fn is_ready(&self) -> bool {
        self.ready
    }
    fn len(&self) -> usize {
        0
    }

    async fn add(&mut self, records: Vec<UtteranceRecord>) -> Result<()> {
        for r in records {
            let emb = format!(
                "[{}]",
                r.embedding
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let meta = serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".into());
            let schemas = r
                .function_schemas
                .as_ref()
                .and_then(|s| serde_json::to_string(s).ok());
            let t = validate_ident(&self.table)?;
            let sql = format!(
                "INSERT INTO {t} (route, utterance, embedding, metadata, function_schemas)
                 VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb)"
            );
            self.client
                .execute(
                    sql.as_str(),
                    &[&r.route, &r.utterance, &emb, &meta, &schemas],
                )
                .await
                .map_err(|e| Error::Database(e.to_string()))?;
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
        let emb = format!(
            "[{}]",
            vector
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let t = validate_ident(&self.table)?;
        let fetch_k = if route_filter.is_some() {
            (top_k as i64).saturating_mul(20).max(top_k as i64)
        } else {
            top_k as i64
        };
        let sql = format!(
            "SELECT route, 1 - (embedding <=> $1::vector) AS score
             FROM {t} ORDER BY embedding <=> $1::vector LIMIT $2"
        );
        let rows = self
            .client
            .query(sql.as_str(), &[&emb, &fetch_k])
            .await
            .map_err(|e| Error::Database(e.to_string()))?;
        let mut scores = Vec::new();
        let mut routes = Vec::new();
        for row in rows {
            let route: String = row.get(0);
            if let Some(filter) = route_filter {
                if !filter.iter().any(|f| f == &route) {
                    continue;
                }
            }
            routes.push(route);
            let s: f64 = row.get(1);
            scores.push(s as f32);
            if routes.len() >= top_k {
                break;
            }
        }
        Ok((scores, routes))
    }

    async fn delete_route(&mut self, route_name: &str) -> Result<()> {
        let t = validate_ident(&self.table)?;
        let sql = format!("DELETE FROM {t} WHERE route = $1");
        self.client
            .execute(sql.as_str(), &[&route_name])
            .await
            .map_err(|e| Error::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_utterances(&self, _include_metadata: bool) -> Result<Vec<Utterance>> {
        let t = validate_ident(&self.table)?;
        let sql = format!("SELECT route, utterance FROM {t}");
        let rows = self
            .client
            .query(sql.as_str(), &[])
            .await
            .map_err(|e| Error::Database(e.to_string()))?;
        Ok(rows
            .iter()
            .map(|r| Utterance::new(r.get::<_, String>(0), r.get::<_, String>(1)))
            .collect())
    }

    async fn clear(&mut self) -> Result<()> {
        let t = validate_ident(&self.table)?;
        let sql = format!("TRUNCATE {t}");
        self.client
            .batch_execute(&sql)
            .await
            .map_err(|e| Error::Database(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::validate_ident;

    #[test]
    fn ident_allows_table_names() {
        assert!(validate_ident("routes").is_ok());
        assert!(validate_ident("intent_routes").is_ok());
    }

    #[test]
    fn ident_rejects_injection() {
        assert!(validate_ident("routes; DROP TABLE x").is_err());
        assert!(validate_ident("").is_err());
        assert!(validate_ident("1routes").is_err());
    }
}
