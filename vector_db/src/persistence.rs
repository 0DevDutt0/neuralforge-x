//! Columnar persistence for the [`VectorStore`] via Apache Parquet.
//!
//! A snapshot is the store's *live* set written as three columns — `id`
//! (`UInt64`), `vector` (`List<Float32>`), and `metadata` (a JSON `Utf8`
//! string) — using the pure-Rust `arrow`/`parquet` stack, so it builds without a
//! bundled C++ toolchain. The graph itself is **not** serialised: it is rebuilt
//! by replaying the vectors through [`VectorStore::insert`] on load. That keeps
//! the on-disk format engine-agnostic and trivially queryable by external tools —
//! e.g. DuckDB can `SELECT … FROM 'snapshot.parquet'` and run its own
//! `array_cosine_similarity` over the `vector` column directly.
//!
//! Index parameters (metric, dimensionality, `M`, `ef`, seed) travel in the
//! Parquet file-level key/value metadata so a snapshot is self-describing.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float32Builder, ListArray, ListBuilder, StringArray, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use neuralforge_core::Metric;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::format::KeyValue;

use crate::error::{Result, VectorDbError};
use crate::hnsw::HnswConfig;
use crate::store::VectorStore;

/// Wraps any persistence-layer failure as [`VectorDbError::Persistence`].
fn fail(e: impl std::fmt::Display) -> VectorDbError {
    VectorDbError::Persistence(e.to_string())
}

/// Writes a snapshot of `store`'s live vectors to a Parquet file at `path`.
///
/// # Errors
/// [`VectorDbError::Persistence`] if the file cannot be created or encoded.
pub fn save(store: &VectorStore, path: impl AsRef<Path>) -> Result<()> {
    let mut ids: Vec<u64> = Vec::with_capacity(store.len());
    let mut metas: Vec<String> = Vec::with_capacity(store.len());
    let mut vectors = ListBuilder::new(Float32Builder::new());

    for (id, vector, metadata) in store.live_records() {
        ids.push(id);
        metas.push(serde_json::to_string(metadata).map_err(fail)?);
        vectors.values().append_slice(vector);
        vectors.append(true);
    }

    let id_arr = Arc::new(UInt64Array::from(ids)) as ArrayRef;
    let vector_arr = Arc::new(vectors.finish()) as ArrayRef;
    let meta_arr = Arc::new(StringArray::from(metas)) as ArrayRef;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", id_arr),
        ("vector", vector_arr),
        ("metadata", meta_arr),
    ])
    .map_err(fail)?;

    let config = store.config();
    let kv = vec![
        KeyValue::new("nfx.metric".to_owned(), store.metric().as_str().to_owned()),
        KeyValue::new("nfx.dim".to_owned(), store.dim().to_string()),
        KeyValue::new("nfx.m".to_owned(), config.m.to_string()),
        KeyValue::new("nfx.m0".to_owned(), config.m0.to_string()),
        KeyValue::new(
            "nfx.ef_construction".to_owned(),
            config.ef_construction.to_string(),
        ),
        KeyValue::new("nfx.ef_search".to_owned(), config.ef_search.to_string()),
        KeyValue::new("nfx.seed".to_owned(), config.seed.to_string()),
    ];
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    let file = File::create(path).map_err(fail)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).map_err(fail)?;
    writer.write(&batch).map_err(fail)?;
    writer.close().map_err(fail)?;
    Ok(())
}

/// Loads a store from a Parquet snapshot, rebuilding the HNSW graph by replaying
/// the persisted vectors.
///
/// The metric and index parameters are read from the file's key/value metadata,
/// so the reconstructed store matches the one that was saved.
///
/// # Errors
/// [`VectorDbError::Persistence`] if the file cannot be read or is malformed.
pub fn load(path: impl AsRef<Path>) -> Result<VectorStore> {
    let file = File::open(path).map_err(fail)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(fail)?;

    let kv = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    let get = |key: &str| {
        kv.iter()
            .find(|k| k.key == key)
            .and_then(|k| k.value.clone())
    };
    let parse = |key: &str| -> Result<usize> {
        get(key)
            .ok_or_else(|| fail(format!("snapshot missing metadata key '{key}'")))?
            .parse::<usize>()
            .map_err(fail)
    };

    let metric = get("nfx.metric")
        .and_then(|s| Metric::from_name(&s))
        .ok_or_else(|| fail("snapshot has an unknown or missing metric"))?;
    let dim = parse("nfx.dim")?;

    let mut config = HnswConfig::new(metric);
    config.m = parse("nfx.m")?;
    config.m0 = parse("nfx.m0")?;
    config.ef_construction = parse("nfx.ef_construction")?;
    config.ef_search = parse("nfx.ef_search")?;
    config.seed = get("nfx.seed")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(config.seed);

    let mut store = VectorStore::with_config(dim, config);

    let reader = builder.build().map_err(fail)?;
    for batch in reader {
        let batch = batch.map_err(fail)?;
        let ids = downcast::<UInt64Array>(&batch, "id")?;
        let vectors = downcast::<ListArray>(&batch, "vector")?;
        let metas = downcast::<StringArray>(&batch, "metadata")?;

        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            let list = vectors.value(row);
            let floats = list
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| fail("vector column is not List<Float32>"))?;
            let vector = floats.values().to_vec();
            let metadata = serde_json::from_str(metas.value(row)).map_err(fail)?;
            store.insert(id, &vector, metadata)?;
        }
    }
    Ok(store)
}

/// Downcasts a named column of `batch` to a concrete Arrow array type.
fn downcast<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .ok_or_else(|| fail(format!("snapshot missing column '{name}'")))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| fail(format!("column '{name}' has an unexpected type")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Filter, MetaValue, Metadata};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nfx_vdb_{}_{}.parquet", name, std::process::id()));
        p
    }

    #[test]
    fn round_trips_vectors_metadata_and_filters() {
        let mut s = VectorStore::new(3, Metric::Cosine);
        for i in 0..50u64 {
            let v = vec![i as f32, (i % 7) as f32, 1.0];
            let md: Metadata = [("g".to_owned(), MetaValue::Int((i % 3) as i64))]
                .into_iter()
                .collect();
            s.insert(i, &v, md).unwrap();
        }
        s.delete(7).unwrap(); // tombstone must not be persisted

        let path = temp_path("roundtrip");
        save(&s, &path).unwrap();
        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), s.len());
        assert!(!loaded.contains(7));
        assert_eq!(loaded.metric(), Metric::Cosine);
        assert_eq!(loaded.dim(), 3);

        // The reloaded index still searches and filters.
        let f = Filter::Eq("g".into(), MetaValue::Int(1));
        let hits = loaded.search(&[10.0, 3.0, 1.0], 5, 0, Some(&f)).unwrap();
        assert!(!hits.is_empty());
        for h in &hits {
            assert_eq!(
                loaded.metadata(h.id).unwrap().get("g"),
                Some(&MetaValue::Int(1))
            );
        }
    }

    #[test]
    fn empty_store_round_trips() {
        let s = VectorStore::new(4, Metric::L2);
        let path = temp_path("empty");
        save(&s, &path).unwrap();
        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(loaded.is_empty());
        assert_eq!(loaded.metric(), Metric::L2);
        assert_eq!(loaded.dim(), 4);
    }
}
