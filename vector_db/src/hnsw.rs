//! A hand-written Hierarchical Navigable Small World (HNSW) graph index.
//!
//! HNSW (Malkov & Yashunin, 2016) is a multi-layer proximity graph that answers
//! approximate nearest-neighbour queries in roughly `O(log N)` hops. Each node is
//! assigned an exponentially-thinning maximum layer; the sparse upper layers act
//! as an express skip-list that funnels a greedy walk toward the query's
//! neighbourhood, and the dense layer 0 (which contains every node) does the fine
//! search. This module implements the graph, its construction heuristic, and a
//! filtered beam search from scratch — the only borrowed machinery is the SIMD
//! distance kernel from [`neuralforge_core`], so the graph and any brute-force
//! ground truth share one numeric implementation.
//!
//! ## What lives here vs. in the store
//!
//! The graph is keyed by dense internal node ids (`0..len`). It is intentionally
//! *append-only*: deletions and updates are modelled by the [`crate::store`]
//! layer as tombstones plus re-insertion, which keeps the graph invariants simple
//! and the hot insert path allocation-light. Result admissibility — tombstones
//! and metadata filters alike — is expressed through an `admit` predicate handed
//! to [`Hnsw::search`]; non-admitted nodes still route the walk (so the graph
//! stays connected) but never enter the result set.

use std::collections::{BinaryHeap, HashSet};

use neuralforge_core::{simd, Metric, Neighbor};

/// Tuning parameters for an [`Hnsw`] graph.
///
/// The defaults (`m = 16`, `ef_construction = 200`, `ef_search = 64`) are the
/// usual mid-range operating point: good recall at a few hundred bytes of graph
/// overhead per vector. `m0` defaults to `2 · m`, following the paper's
/// recommendation that the base layer be twice as dense as the upper layers.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Target out-degree on layers above 0.
    pub m: usize,
    /// Maximum out-degree on layer 0 (defaults to `2 · m`).
    pub m0: usize,
    /// Size of the dynamic candidate list while *building* the graph. Larger
    /// values spend more time per insert for a denser, higher-recall graph.
    pub ef_construction: usize,
    /// Default size of the dynamic candidate list while *searching*. May be
    /// overridden per query; the search clamps it up to at least `k`.
    pub ef_search: usize,
    /// The metric the graph is ordered by.
    pub metric: Metric,
    /// Seed for the level-assignment PRNG, so construction is reproducible.
    pub seed: u64,
}

impl HnswConfig {
    /// A configuration with the given metric and the default graph parameters.
    #[must_use]
    pub fn new(metric: Metric) -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
            metric,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Sets `m` (and `m0 = 2·m`), returning `self` for chaining.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m.max(2);
        self.m0 = self.m * 2;
        self
    }

    /// Sets `ef_construction`, returning `self` for chaining.
    #[must_use]
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef.max(1);
        self
    }

    /// Sets the default `ef_search`, returning `self` for chaining.
    #[must_use]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef.max(1);
        self
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(Metric::Cosine)
    }
}

/// A `(distance, node)` pair ordered by distance, smallest distance "best".
///
/// A `BinaryHeap<Candidate>` is therefore a max-heap whose top is the *farthest*
/// element (handy as the dynamic result list, where we evict the worst), while a
/// `BinaryHeap<Reverse<Candidate>>` is a min-heap whose top is the *nearest*
/// (the exploration frontier).
#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f32,
    node: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    /// Total order on `(dist, node)`. `total_cmp` keeps `NaN` from poisoning the
    /// heap; the `node` tie-break makes traversal deterministic.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.node.cmp(&other.node))
    }
}

/// A fast, allocation-free `splitmix64` PRNG used only for level assignment.
///
/// HNSW needs a stream of uniforms to draw each node's layer; seeding it (rather
/// than pulling from the OS) makes graph construction bit-for-bit reproducible,
/// which the recall tests rely on.
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform in the half-open interval `(0, 1)`, drawn from 24 random bits.
    #[inline]
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 40; // top 24 bits
        (bits as f64 + 1.0) / ((1u64 << 24) as f64 + 1.0)
    }
}

/// A hierarchical navigable small-world graph over dense `f32` vectors.
pub struct Hnsw {
    dim: usize,
    config: HnswConfig,
    /// Row-major vector store; node `i` occupies `data[i*dim .. (i+1)*dim]`. For
    /// the cosine metric vectors are L2-normalised on insertion, so an inner
    /// product *is* the cosine similarity and no per-query norm is needed.
    data: Vec<f32>,
    /// `links[i][lc]` is node `i`'s adjacency list on layer `lc`. Node `i` exists
    /// on layers `0..=links[i].len()-1`.
    links: Vec<Vec<Vec<usize>>>,
    entry: Option<usize>,
    max_level: usize,
    rng: SplitMix64,
    /// Level-generation normaliser, `1 / ln(m)`.
    ml: f64,
}

impl Hnsw {
    /// Creates an empty index for `dim`-dimensional vectors.
    #[must_use]
    pub fn new(dim: usize, config: HnswConfig) -> Self {
        let ml = 1.0 / (config.m.max(2) as f64).ln();
        Self {
            dim,
            config,
            data: Vec::new(),
            links: Vec::new(),
            entry: None,
            max_level: 0,
            rng: SplitMix64(config.seed),
            ml,
        }
    }

    /// The dimensionality of vectors in this index.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of nodes ever inserted (including tombstoned ones the store
    /// has logically deleted).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Whether the graph contains no nodes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// The configured metric.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.config.metric
    }

    /// The graph's configuration (used when rebuilding/compacting the store).
    #[inline]
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        self.config
    }

    /// Borrows node `i`'s stored vector (normalised for cosine). Exposed for the
    /// store's snapshot and compaction paths.
    #[inline]
    #[must_use]
    pub fn raw_vector(&self, i: usize) -> &[f32] {
        self.vector(i)
    }

    /// Borrows node `i`'s stored vector.
    #[inline]
    fn vector(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Internal distance from a prepared query `q` to node `i`, where *smaller is
    /// nearer* regardless of metric. For cosine the stored vectors are unit-norm,
    /// so `1 − ⟨q, v⟩` is the cosine distance; dot-product is negated so that
    /// "more similar" sorts first; L2 uses the squared distance (monotone in the
    /// true distance, so ordering is identical but cheaper).
    #[inline]
    fn dist_to(&self, q: &[f32], i: usize) -> f32 {
        let v = self.vector(i);
        match self.config.metric {
            Metric::Cosine => 1.0 - simd::dot(q, v),
            Metric::DotProduct => -simd::dot(q, v),
            Metric::L2 => simd::l2_sq(q, v),
        }
    }

    /// Internal distance between two stored nodes (used by the build heuristic).
    #[inline]
    fn dist_nodes(&self, a: usize, b: usize) -> f32 {
        let va = self.vector(a);
        let vb = self.vector(b);
        match self.config.metric {
            Metric::Cosine => 1.0 - simd::dot(va, vb),
            Metric::DotProduct => -simd::dot(va, vb),
            Metric::L2 => simd::l2_sq(va, vb),
        }
    }

    /// Converts an internal distance back into the metric's reported score:
    /// cosine/dot similarity (higher better) or true L2 distance (lower better).
    #[inline]
    fn to_score(&self, dist: f32) -> f32 {
        match self.config.metric {
            Metric::Cosine => 1.0 - dist,
            Metric::DotProduct => -dist,
            Metric::L2 => dist.max(0.0).sqrt(),
        }
    }

    /// Prepares a raw query for traversal: L2-normalised for cosine, copied as-is
    /// otherwise (a zero-norm vector is left untouched and will score `0`).
    fn prepare(&self, q: &[f32]) -> Vec<f32> {
        match self.config.metric {
            Metric::Cosine => {
                let norm = simd::norm_sq(q).sqrt();
                if norm == 0.0 {
                    q.to_vec()
                } else {
                    q.iter().map(|x| x / norm).collect()
                }
            }
            _ => q.to_vec(),
        }
    }

    /// Draws a node's top layer from the geometric-ish HNSW distribution
    /// `floor(-ln(U) · ml)`.
    fn random_level(&mut self) -> usize {
        let u = self.rng.next_unit();
        (-u.ln() * self.ml).floor() as usize
    }

    /// Inserts a vector and returns its internal node id.
    ///
    /// The caller (the store) is responsible for validating dimensionality and
    /// finiteness; in debug builds the dimension is asserted here too.
    pub fn insert(&mut self, vector: &[f32]) -> usize {
        debug_assert_eq!(vector.len(), self.dim, "vector width must equal index dim");
        let id = self.len();
        let level = self.random_level();

        // Store the (normalised, for cosine) vector and allocate its link rows.
        let prepared = self.prepare(vector);
        self.data.extend_from_slice(&prepared);
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            // First node becomes the entry point at its own level.
            self.entry = Some(id);
            self.max_level = level;
            return id;
        };

        let q = prepared; // owned; avoids borrowing `self.data` during mutation
        let mut visited = HashSet::new();

        // Phase 1 — greedy descent through the express layers with ef = 1.
        let mut ep = vec![entry];
        let top = self.max_level;
        for lc in ((level + 1)..=top).rev() {
            visited.clear();
            let w = self.search_layer(&q, &ep, 1, lc, &|_| true, &mut visited);
            if let Some(best) = w.first() {
                ep = vec![best.node];
            }
        }

        // Phase 2 — for each layer the node lives on, find ef_construction
        // neighbours, prune to M with the heuristic, and wire bidirectional links.
        let start = level.min(top);
        for lc in (0..=start).rev() {
            visited.clear();
            let w = self.search_layer(
                &q,
                &ep,
                self.config.ef_construction,
                lc,
                &|_| true,
                &mut visited,
            );
            let m = if lc == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected = self.select_neighbors(&w, m);

            for &nbr in &selected {
                self.links[id][lc].push(nbr);
                self.links[nbr][lc].push(id);
                if self.links[nbr][lc].len() > m {
                    self.prune(nbr, lc, m);
                }
            }
            ep = w.iter().map(|c| c.node).collect();
        }

        if level > self.max_level {
            self.entry = Some(id);
            self.max_level = level;
        }
        id
    }

    /// The dynamic-list beam search of one layer.
    ///
    /// Explores from `entry` along layer `level`, keeping the `ef` nearest
    /// *admitted* nodes in a bounded result heap while the full visited frontier
    /// drives exploration. Returns the result list sorted nearest-first. The
    /// `visited` set is supplied by the caller so it can be reused (cleared)
    /// across the layers of a single query without reallocating.
    fn search_layer(
        &self,
        q: &[f32],
        entry: &[usize],
        ef: usize,
        level: usize,
        admit: &impl Fn(usize) -> bool,
        visited: &mut HashSet<usize>,
    ) -> Vec<Candidate> {
        // Min-heap on distance: the next node to expand is always the closest.
        let mut frontier: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();
        // Max-heap on distance: its top is the worst admitted result, evicted first.
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &e in entry {
            if visited.insert(e) {
                let c = Candidate {
                    dist: self.dist_to(q, e),
                    node: e,
                };
                frontier.push(std::cmp::Reverse(c));
                if admit(e) {
                    results.push(c);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        while let Some(std::cmp::Reverse(c)) = frontier.pop() {
            // Stop once the frontier's nearest is farther than the worst result
            // and we already hold a full set of admitted candidates.
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if c.dist > worst.dist {
                        break;
                    }
                }
            }
            for &nb in &self.links[c.node][level] {
                if visited.insert(nb) {
                    let d = self.dist_to(q, nb);
                    let room = results.len() < ef;
                    let improves = results.peek().map_or(true, |w| d < w.dist);
                    if room || improves {
                        frontier.push(std::cmp::Reverse(Candidate { dist: d, node: nb }));
                        if admit(nb) {
                            results.push(Candidate { dist: d, node: nb });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results.into_sorted_vec() // ascending by distance → nearest first
    }

    /// The neighbour-selection heuristic (paper Algorithm 4).
    ///
    /// Walks the `ef` candidates nearest-first and keeps an edge to `e` only if
    /// `e` is closer to the new node than to every neighbour already chosen — this
    /// favours edges that reach *new* regions of space over a cluster of mutually
    /// close points, which is what gives HNSW its long-range navigability. Slots
    /// left unfilled are then backfilled with the closest discarded candidates so
    /// the degree target is still met.
    fn select_neighbors(&self, candidates: &[Candidate], m: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let mut discarded: Vec<usize> = Vec::new();

        for c in candidates {
            if selected.len() >= m {
                break;
            }
            let closer_to_query_than_to_peers = selected
                .iter()
                .all(|&r| self.dist_nodes(c.node, r) >= c.dist);
            if closer_to_query_than_to_peers {
                selected.push(c.node);
            } else {
                discarded.push(c.node);
            }
        }

        // Keep pruned connections: fill remaining slots with the nearest leftovers.
        let mut di = 0;
        while selected.len() < m && di < discarded.len() {
            selected.push(discarded[di]);
            di += 1;
        }
        selected
    }

    /// Re-applies the selection heuristic to an over-full adjacency list,
    /// trimming node `node`'s layer-`level` neighbours back to `m`.
    fn prune(&mut self, node: usize, level: usize, m: usize) {
        let mut cands: Vec<Candidate> = self.links[node][level]
            .iter()
            .map(|&nb| Candidate {
                dist: self.dist_nodes(node, nb),
                node: nb,
            })
            .collect();
        cands.sort_unstable();
        let kept = self.select_neighbors(&cands, m);
        self.links[node][level] = kept;
    }

    /// Searches the graph for the `k` nearest admitted neighbours of `query`.
    ///
    /// `ef` is the search-time dynamic-list size; it is clamped up to at least
    /// `k`. The `admit` predicate gates *result membership* only — non-admitted
    /// nodes (tombstones, filtered-out metadata) still route the walk. Fewer than
    /// `k` results may come back when the admit predicate is highly selective.
    ///
    /// Scores are reported in the metric's natural units (cosine/dot similarity,
    /// or true L2 distance), best match first.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        admit: &impl Fn(usize) -> bool,
    ) -> Vec<Neighbor> {
        if self.entry.is_none() || k == 0 {
            return Vec::new();
        }
        let q = self.prepare(query);
        let mut visited = HashSet::new();

        // Greedy descent through the express layers (unfiltered, ef = 1).
        let mut ep = vec![self.entry.unwrap()];
        for lc in (1..=self.max_level).rev() {
            visited.clear();
            let w = self.search_layer(&q, &ep, 1, lc, &|_| true, &mut visited);
            if let Some(best) = w.first() {
                ep = vec![best.node];
            }
        }

        // Fine, filtered search of the base layer.
        visited.clear();
        let ef = ef.max(self.config.ef_search).max(k);
        let w = self.search_layer(&q, &ep, ef, 0, admit, &mut visited);
        w.into_iter()
            .take(k)
            .map(|c| Neighbor {
                index: c.node,
                score: self.to_score(c.dist),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(dim: usize, rows: &[Vec<f32>], metric: Metric) -> Hnsw {
        let mut h = Hnsw::new(dim, HnswConfig::new(metric));
        for r in rows {
            h.insert(r);
        }
        h
    }

    #[test]
    fn finds_exact_match_first() {
        let rows = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![-1.0, 0.0],
        ];
        let h = build(2, &rows, Metric::Cosine);
        let res = h.search(&[1.0, 0.0], 2, 32, &|_| true);
        assert_eq!(res[0].index, 0);
        assert!((res[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn admit_predicate_excludes_tombstones() {
        let rows = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
        let h = build(2, &rows, Metric::Cosine);
        // Exclude the exact match; the next-best should lead.
        let res = h.search(&[1.0, 0.0], 1, 32, &|n| n != 0);
        assert_eq!(res[0].index, 1);
    }

    #[test]
    fn empty_index_returns_nothing() {
        let h = Hnsw::new(4, HnswConfig::default());
        assert!(h.is_empty());
        assert!(h.search(&[0.0; 4], 5, 16, &|_| true).is_empty());
    }

    #[test]
    fn l2_reports_true_distance() {
        let rows = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let h = build(2, &rows, Metric::L2);
        let res = h.search(&[0.0, 0.0], 2, 16, &|_| true);
        assert_eq!(res[0].index, 0);
        assert!(res[0].score < 1e-5);
        // Distance to (3,4) is exactly 5.
        assert!((res[1].score - 5.0).abs() < 1e-4);
    }
}
