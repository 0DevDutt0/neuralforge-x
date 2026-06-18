//! The vector store — the repository surface over the [`Hnsw`] graph.
//!
//! The graph speaks in dense internal node ids and is append-only; this layer
//! turns it into a mutable, id-addressed store. It owns the bidirectional map
//! between caller-chosen `u64` ids and internal nodes, the per-vector
//! [`Metadata`], and a tombstone set. Deletes flip a tombstone; updates that
//! change the vector tombstone the old node and re-insert (keeping the same
//! external id). Tombstones are skipped at query time via the graph's `admit`
//! predicate and physically removed by [`VectorStore::compact`].

use std::collections::HashMap;

use neuralforge_core::Metric;

use crate::error::{Result, VectorDbError};
use crate::hnsw::{Hnsw, HnswConfig};
use crate::metadata::{Filter, Metadata};

/// A single search result: the caller's id and the metric score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The external id supplied at insertion time.
    pub id: u64,
    /// Metric value — cosine/dot similarity (higher better) or L2 distance
    /// (lower better), matching the store's configured metric.
    pub score: f32,
}

/// An HNSW-backed vector store with metadata and soft deletes.
pub struct VectorStore {
    index: Hnsw,
    /// Live external id → internal node. Tombstoned ids are absent.
    id_to_node: HashMap<u64, usize>,
    /// Internal node → external id, for *every* node (including tombstones), so
    /// it stays index-aligned with the graph.
    node_to_id: Vec<u64>,
    /// Per-node metadata, index-aligned with the graph.
    metadata: Vec<Metadata>,
    /// Per-node tombstone flag, index-aligned with the graph.
    deleted: Vec<bool>,
    /// Count of non-tombstoned nodes.
    live: usize,
}

impl VectorStore {
    /// Creates an empty store for `dim`-dimensional vectors under `metric`.
    #[must_use]
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self::with_config(dim, HnswConfig::new(metric))
    }

    /// Creates an empty store with a fully specified HNSW configuration.
    #[must_use]
    pub fn with_config(dim: usize, config: HnswConfig) -> Self {
        Self {
            index: Hnsw::new(dim, config),
            id_to_node: HashMap::new(),
            node_to_id: Vec::new(),
            metadata: Vec::new(),
            deleted: Vec::new(),
            live: 0,
        }
    }

    /// Vector dimensionality.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.index.dim()
    }

    /// The metric the store ranks by.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.index.metric()
    }

    /// The underlying HNSW configuration (used by the persistence layer).
    #[inline]
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        self.index.config()
    }

    /// Number of live (non-deleted) vectors.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether the store holds no live vectors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Whether `id` is currently live in the store.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_node.contains_key(&id)
    }

    /// Borrows the metadata for a live `id`, if present.
    #[must_use]
    pub fn metadata(&self, id: u64) -> Option<&Metadata> {
        self.id_to_node.get(&id).map(|&n| &self.metadata[n])
    }

    /// Validates a vector against the store's dimension and finiteness contract.
    fn validate(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim() {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.dim(),
                actual: vector.len(),
            });
        }
        if let Some(pos) = vector.iter().position(|x| !x.is_finite()) {
            return Err(VectorDbError::NonFinite { pos });
        }
        Ok(())
    }

    /// Inserts a new vector under a fresh external `id`.
    ///
    /// # Errors
    /// [`VectorDbError::DuplicateId`] if `id` is already live,
    /// [`VectorDbError::DimensionMismatch`] on a width mismatch, or
    /// [`VectorDbError::NonFinite`] if the vector contains `NaN`/`±∞`.
    pub fn insert(&mut self, id: u64, vector: &[f32], metadata: Metadata) -> Result<()> {
        if self.id_to_node.contains_key(&id) {
            return Err(VectorDbError::DuplicateId { id });
        }
        self.validate(vector)?;
        let node = self.index.insert(vector);
        debug_assert_eq!(
            node,
            self.node_to_id.len(),
            "store vectors fell out of sync"
        );
        self.node_to_id.push(id);
        self.metadata.push(metadata);
        self.deleted.push(false);
        self.id_to_node.insert(id, node);
        self.live += 1;
        Ok(())
    }

    /// Soft-deletes a live `id`, tombstoning its node.
    ///
    /// # Errors
    /// [`VectorDbError::UnknownId`] if `id` is not live.
    pub fn delete(&mut self, id: u64) -> Result<()> {
        let node = self
            .id_to_node
            .remove(&id)
            .ok_or(VectorDbError::UnknownId { id })?;
        self.deleted[node] = true;
        self.live -= 1;
        Ok(())
    }

    /// Updates a live `id`'s vector and/or metadata.
    ///
    /// A new `vector` tombstones the old node and re-inserts (rewiring the graph)
    /// under the same id, carrying the existing metadata forward unless new
    /// metadata is also supplied. A metadata-only update mutates in place.
    ///
    /// # Errors
    /// [`VectorDbError::UnknownId`] if `id` is not live, plus the validation
    /// errors of [`VectorStore::insert`] when a vector is supplied.
    pub fn update(
        &mut self,
        id: u64,
        vector: Option<&[f32]>,
        metadata: Option<Metadata>,
    ) -> Result<()> {
        let node = *self
            .id_to_node
            .get(&id)
            .ok_or(VectorDbError::UnknownId { id })?;

        match vector {
            Some(v) => {
                self.validate(v)?;
                let carried = metadata.unwrap_or_else(|| self.metadata[node].clone());
                // Tombstone the old node, then insert the replacement.
                self.deleted[node] = true;
                let new_node = self.index.insert(v);
                self.node_to_id.push(id);
                self.metadata.push(carried);
                self.deleted.push(false);
                self.id_to_node.insert(id, new_node);
                // live count is unchanged: one out, one in.
            }
            None => {
                if let Some(md) = metadata {
                    self.metadata[node] = md;
                }
            }
        }
        Ok(())
    }

    /// Returns the `k` nearest live vectors to `query`, optionally constrained by
    /// a metadata `filter`, best match first.
    ///
    /// `ef` is the search beam width; pass `0` to use the configured default.
    /// With a selective `filter`, fewer than `k` hits may be returned.
    ///
    /// # Errors
    /// [`VectorDbError::InvalidK`] if `k == 0` or `k` exceeds the live count, and
    /// the validation errors of [`VectorStore::insert`] for the query vector.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        self.validate(query)?;
        if k == 0 || k > self.live {
            return Err(VectorDbError::InvalidK { k, live: self.live });
        }

        let admit = |node: usize| -> bool {
            !self.deleted[node] && filter.map_or(true, |f| f.matches(&self.metadata[node]))
        };

        let neighbors = self.index.search(query, k, ef, &admit);
        Ok(neighbors
            .into_iter()
            .map(|n| Hit {
                id: self.node_to_id[n.index],
                score: n.score,
            })
            .collect())
    }

    /// Number of tombstoned nodes still occupying graph memory.
    #[must_use]
    pub fn tombstones(&self) -> usize {
        self.node_to_id.len() - self.live
    }

    /// Physically removes tombstones by rebuilding the graph from the live
    /// vectors, reclaiming their memory and the dead edges that pointed at them.
    ///
    /// Internal node ids are reassigned; external ids and metadata are preserved.
    pub fn compact(&mut self) {
        if self.tombstones() == 0 {
            return;
        }
        let mut rebuilt = VectorStore::with_config(self.dim(), self.index.config());
        for node in 0..self.node_to_id.len() {
            if self.deleted[node] {
                continue;
            }
            let id = self.node_to_id[node];
            let vector = self.index.raw_vector(node).to_vec();
            let md = std::mem::take(&mut self.metadata[node]);
            // Re-insertion of already-validated vectors cannot fail.
            let _ = rebuilt.insert(id, &vector, md);
        }
        *self = rebuilt;
    }

    /// An iterator over every live `(id, vector, metadata)` triple, in internal
    /// node order. Used by the persistence layer to snapshot the store.
    pub(crate) fn live_records(&self) -> impl Iterator<Item = (u64, &[f32], &Metadata)> {
        (0..self.node_to_id.len()).filter_map(move |node| {
            if self.deleted[node] {
                None
            } else {
                Some((
                    self.node_to_id[node],
                    self.index.raw_vector(node),
                    &self.metadata[node],
                ))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{MetaValue, Metadata};

    fn md(pairs: &[(&str, MetaValue)]) -> Metadata {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect()
    }

    fn store() -> VectorStore {
        let mut s = VectorStore::new(2, Metric::Cosine);
        s.insert(10, &[1.0, 0.0], md(&[("tag", "a".into())]))
            .unwrap();
        s.insert(20, &[0.0, 1.0], md(&[("tag", "b".into())]))
            .unwrap();
        s.insert(30, &[1.0, 1.0], md(&[("tag", "a".into())]))
            .unwrap();
        s
    }

    #[test]
    fn insert_and_search_returns_external_ids() {
        let s = store();
        let hits = s.search(&[1.0, 0.05], 2, 0, None).unwrap();
        assert_eq!(hits[0].id, 10);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut s = store();
        assert!(matches!(
            s.insert(10, &[0.5, 0.5], Metadata::new()),
            Err(VectorDbError::DuplicateId { id: 10 })
        ));
    }

    #[test]
    fn delete_then_excluded_from_results() {
        let mut s = store();
        s.delete(10).unwrap();
        assert_eq!(s.len(), 2);
        assert!(!s.contains(10));
        let hits = s.search(&[1.0, 0.0], 2, 0, None).unwrap();
        assert!(hits.iter().all(|h| h.id != 10));
    }

    #[test]
    fn metadata_filter_restricts_results() {
        let s = store();
        let f = Filter::Eq("tag".into(), "a".into());
        let hits = s.search(&[0.0, 1.0], 2, 0, Some(&f)).unwrap();
        assert!(hits.iter().all(|h| h.id == 10 || h.id == 30));
    }

    #[test]
    fn update_vector_keeps_id_and_metadata() {
        let mut s = store();
        s.update(20, Some(&[1.0, 0.02]), None).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(
            s.metadata(20).unwrap().get("tag"),
            Some(&MetaValue::from("b"))
        );
        let hits = s.search(&[1.0, 0.0], 1, 0, None).unwrap();
        // 20 now points almost along +x; still one of the top matches.
        assert!(hits[0].id == 10 || hits[0].id == 20);
    }

    #[test]
    fn compact_drops_tombstones() {
        let mut s = store();
        s.delete(20).unwrap();
        assert_eq!(s.tombstones(), 1);
        s.compact();
        assert_eq!(s.tombstones(), 0);
        assert_eq!(s.len(), 2);
        // Surviving ids still searchable after the rebuild.
        let hits = s.search(&[1.0, 1.0], 2, 0, None).unwrap();
        assert!(hits.iter().any(|h| h.id == 30));
    }

    #[test]
    fn unknown_id_errors() {
        let mut s = store();
        assert!(matches!(
            s.delete(999),
            Err(VectorDbError::UnknownId { id: 999 })
        ));
    }

    #[test]
    fn invalid_k_rejected() {
        let s = store();
        assert!(matches!(
            s.search(&[1.0, 0.0], 0, 0, None),
            Err(VectorDbError::InvalidK { k: 0, live: 3 })
        ));
        assert!(matches!(
            s.search(&[1.0, 0.0], 99, 0, None),
            Err(VectorDbError::InvalidK { k: 99, live: 3 })
        ));
    }
}
