//! Utterance diff / sync strategy — port of schema.UtteranceDiff.get_sync_strategy.
//! LocalIndex is a no-op for hash/lock. Remote indexes apply the plan.
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use crate::error::{Error, Result};
use crate::schema::{SyncMode, Utterance};

#[derive(Debug, Clone, Default)]
pub struct SyncPlan {
    pub remote_upsert: Vec<Utterance>,
    pub remote_delete: Vec<Utterance>,
    pub local_upsert: Vec<Utterance>,
    pub local_delete: Vec<Utterance>,
}

pub fn diff_and_plan(
    local: &[Utterance],
    remote: &[Utterance],
    mode: SyncMode,
) -> Result<SyncPlan> {
    let local_keys: Vec<String> = {
        let mut v: Vec<_> = local.iter().map(|u| u.to_str(false)).collect();
        v.sort();
        v
    };
    let remote_keys: Vec<String> = {
        let mut v: Vec<_> = remote.iter().map(|u| u.to_str(false)).collect();
        v.sort();
        v
    };
    let local_map: std::collections::BTreeMap<_, _> =
        local.iter().map(|u| (u.to_str(false), u.clone())).collect();
    let remote_map: std::collections::BTreeMap<_, _> = remote
        .iter()
        .map(|u| (u.to_str(false), u.clone()))
        .collect();

    let mut local_only = Vec::new();
    let mut remote_only = Vec::new();
    for k in &local_keys {
        if !remote_map.contains_key(k) {
            local_only.push(local_map[k].clone());
        }
    }
    for k in &remote_keys {
        if !local_map.contains_key(k) {
            remote_only.push(remote_map[k].clone());
        }
    }

    match mode {
        SyncMode::Error => {
            if !local_only.is_empty() || !remote_only.is_empty() {
                return Err(Error::SyncConflict);
            }
            Ok(SyncPlan::default())
        }
        SyncMode::Local => Ok(SyncPlan {
            remote_upsert: local_only,
            remote_delete: remote_only,
            ..Default::default()
        }),
        SyncMode::Remote => Ok(SyncPlan {
            local_upsert: remote_only,
            local_delete: local_only,
            ..Default::default()
        }),
        SyncMode::Merge => Ok(SyncPlan {
            remote_upsert: local_only,
            local_upsert: remote_only,
            ..Default::default()
        }),
        SyncMode::MergeForceLocal => Ok(SyncPlan {
            remote_upsert: local_only,
            remote_delete: remote_only,
            ..Default::default()
        }),
        SyncMode::MergeForceRemote => Ok(SyncPlan {
            local_upsert: remote_only,
            local_delete: local_only,
            ..Default::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_pushes_missing_remote() {
        let local = vec![Utterance::new("a", "hello")];
        let remote = vec![];
        let plan = diff_and_plan(&local, &remote, SyncMode::Local).unwrap();
        assert_eq!(plan.remote_upsert.len(), 1);
        assert!(plan.remote_delete.is_empty());
    }

    #[test]
    fn error_on_drift() {
        let local = vec![Utterance::new("a", "hello")];
        let remote = vec![];
        assert!(diff_and_plan(&local, &remote, SyncMode::Error).is_err());
    }
}
