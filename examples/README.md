# Examples

Runnable demonstrations of the NeuralForge-X API. Activate the project venv
first (or `pwsh scripts/dev_build.ps1` to build the extension).

| Script | What it shows |
|--------|---------------|
| [`01_quickstart.py`](01_quickstart.py) | The five core kernels: pairwise, batch, top-k. |
| [`02_numpy_vs_rust_benchmark.py`](02_numpy_vs_rust_benchmark.py) | NumPy-vs-Rust timings; writes `benchmark_lab/results/python_vs_rust.json`. |
| [`03_vector_db.py`](03_vector_db.py) | HNSW `VectorIndex`: filtered ANN search, Parquet snapshot, and an exact DuckDB SQL baseline over it. |

```bash
python examples/01_quickstart.py
python examples/02_numpy_vs_rust_benchmark.py
python examples/03_vector_db.py   # `pip install duckdb` for the SQL baseline step
```
