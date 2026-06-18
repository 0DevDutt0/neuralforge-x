# vector_db — HNSW vector database (Phase 4)

A Rust vector store implementing approximate nearest-neighbour search over a
hand-written **HNSW** graph, with metadata filtering and columnar persistence.
The graph reuses `neuralforge_core`'s SIMD distance kernels, so the approximate
index and exact brute-force ground truth rank by one identical implementation.

## Repository surface

```text
insert(id, vector, metadata)        delete(id)        update(id, vector?, metadata?)
search(query, k, ef, filter?) -> Vec<Hit>            compact()        save / load (Parquet)
```

```rust
use neuralforge_core::Metric;
use neuralforge_vector_db::{Filter, MetaValue, Metadata, VectorStore};

let mut store = VectorStore::new(3, Metric::Cosine);
store.insert(1, &[1.0, 0.0, 0.0], Metadata::from([("lang".into(), MetaValue::from("rust"))]))?;
let filter = Filter::Eq("lang".into(), "rust".into());
let hits = store.search(&[1.0, 0.05, 0.0], 1, 0, Some(&filter))?;
```

## Design

- **HNSW** — a multi-layer proximity graph (Malkov & Yashunin). Seeded
  `splitmix64` level assignment makes construction reproducible; queries greedily
  descend the sparse express layers, then run an `ef`-bounded beam search of the
  base layer. Edges are chosen with the paper's neighbour-selection heuristic and
  pruned to the degree bound. `M` / `ef_construction` / `ef_search` are tunable.
- **Mutability** — the graph is append-only; `delete`/`update` tombstone nodes
  and `compact()` rebuilds to reclaim memory. A metadata predicate gates *result
  membership* only, so filtered-out and deleted nodes still route the walk and
  recall degrades gracefully with selectivity.
- **Filtering** — a composable [`Filter`] language (`And`/`Or`/`Not`, equality,
  numeric ranges, membership, presence) evaluated during traversal.
- **Persistence** — vectors + metadata in **Parquet** (pure-Rust `arrow`/
  `parquet`), self-describing via file metadata; the index is rebuilt from the
  columnar store on load. The format is engine-agnostic — **DuckDB** can query the
  snapshot directly (see [`examples/03_vector_db.py`](../examples/03_vector_db.py)).

## Validation

Recall@k is measured against brute-force ground truth from `neuralforge_core`
(`tests/recall.rs`); the `criterion` benchmark (`benches/hnsw.rs`) reports build
throughput and the recall-vs-`ef` / latency trade-off.
