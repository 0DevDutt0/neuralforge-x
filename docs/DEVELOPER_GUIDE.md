# Developer Guide

## Prerequisites

- **Rust** (stable). On Windows install the MSVC toolchain: `rustup default stable-msvc`.
  The Visual Studio "Desktop development with C++" workload provides the linker.
- **Python 3.9+** with a virtual environment.
- For Phase 3 (GPU): CUDA Toolkit 12.8+, an `sm_120` driver, PyTorch cu128.

## One-time setup

```bash
git clone https://github.com/0DevDutt0/neuralforge-x && cd neuralforge-x
python -m venv venv
. venv/Scripts/activate          # Windows;  source venv/bin/activate elsewhere
pip install -e ".[dev]"          # builds the extension via the maturin backend
```

## Everyday workflow

```bash
# --- Rust core ---
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo bench -p neuralforge_core            # criterion; HTML in target/criterion

# --- Python SDK ---
python -m pytest
python -m ruff check . && python -m ruff format --check .
python -m mypy
python examples/01_quickstart.py

# --- Vector DB + service (Rust) ---
cargo test -p neuralforge_vector_db
cargo run -p neuralforge_service                       # HTTP service on :8080
cargo clippy -p neuralforge_service --all-targets --features otel -- -D warnings

# --- Benchmark & profiling labs ---
python -m benchmark_lab all                            # cross-stack bench + charts
python profiling/analyze.py                            # CPU profile report + chart

# --- Observability stack (service + Prometheus + Grafana + Jaeger) ---
cd observability && docker compose up --build
```

`make help` lists shortcuts for all of the above.

## Building the extension

Normally `pip install -e .` (or `maturin develop`) builds and installs the
`neuralforge._native` module into your venv.

### Windows: Application Control / Smart App Control

Some Windows 11 machines block the **prebuilt** `maturin.exe` (and other
downloaded PyPI launcher binaries) under an Application Control policy. Locally
*compiled* binaries (cargo's own output) are unaffected. Use the bundled helper,
which builds the cdylib with cargo and installs it without the blocked launcher:

```powershell
pwsh scripts/dev_build.ps1            # cargo build --release -p neuralforge
                                      # + copy target/release/neuralforge_native.dll
                                      #   -> python_sdk/python/neuralforge/_native.pyd
                                      # + write an editable .pth into the venv
```

`ruff` and `mypy` still work because they are invoked as `python -m ruff` /
`python -m mypy` (through the trusted interpreter).

## Project layout

```
rust_core/      kernels (lib) + criterion benches + tests
python_sdk/     src/lib.rs (PyO3) + python/neuralforge/ (typed wrappers, .pyi)
vector_db/      hand-written HNSW index + Parquet persistence (Rust)
cuda_engine/    CUDA C++ / Triton / PyTorch GPU kernels (Python pkg)
observability/  neuralforge_service (axum) + docker-compose stack
benchmark_lab/  cross-stack harness (python -m benchmark_lab) + results/*.json
profiling/      neuralforge_profile target + analyze.py + capture scripts
tests/          Python integration tests (pytest)
examples/       runnable demos
docs/           design docs + docs/assets (generated SVG charts)
scripts/        dev helpers (build, setup)
```

## Testing strategy

- **Rust:** unit tests per module, `proptest` SIMD≡scalar equivalence, an
  integration test (`rust_core/tests/kernels.rs`) cross-checking `top_k_search`
  against `batch_similarity`, and a brute-force argsort oracle.
- **Python:** NumPy parity per kernel/metric, `hypothesis` properties, and a full
  error-path suite (`tests/test_errors.py`).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `link.exe not found` | Install VS Build Tools "Desktop C++"; use `stable-msvc`. |
| `maturin.exe ... blocked` | Use `scripts/dev_build.ps1` (see above). |
| `import neuralforge` fails | Ensure the extension is built and the venv active; re-run `dev_build.ps1`. |
| mypy "3.9 not supported" | mypy targets ≥3.10; runtime still supports 3.9 (abi3). |
| Non-contiguous array error | The wrapper coerces inputs; if calling `_native` directly, pass C-contiguous `float32`. |
| `flatbuffers` build blocked (release) | Application Control flags the build-script; `arrow`/`parquet` use `default-features=false`. If a stale artifact persists, `rm -rf target/release/build/flatbuffers-*` and rebuild. |
| `cargo flamegraph` → `NotAnAdmin` | Windows ETW capture needs an elevated shell; run `profiling/scripts/cpu_flamegraph.ps1` as administrator (the criterion report needs no privileges). |
| Nsight records no CUDA rows | Blackwell `sm_120` kernel tracing needs a newer Nsight than is installed; the scripts are correct for supported toolchains. |
| OTLP traces not arriving | Build the service with `--features otel` and set `NFX_OTEL_ENDPOINT` (e.g. `http://otel-collector:4317`). |
