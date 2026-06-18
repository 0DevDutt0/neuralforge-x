# python_sdk — `neuralforge`

PyO3 bindings + the typed, validated Python API for NeuralForge-X. The compiled
Rust extension is `neuralforge._native`; the public surface is the pure-Python
`neuralforge` package (this is a maturin *mixed* layout).

## Layout

```
python_sdk/
├── Cargo.toml              # cdylib crate (PyO3)
├── src/lib.rs              # #[pymodule] _native — zero-copy NumPy bindings
└── python/neuralforge/
    ├── __init__.py         # typed public API (validation, SearchResult)
    ├── _exceptions.py      # NeuralForgeError hierarchy
    ├── _native.pyi         # type stubs for the extension
    └── py.typed            # PEP 561 marker
```

## Build

From the repo root (maturin backend, configured in the root `pyproject.toml`):

```bash
pip install -e ".[dev]"      # or: maturin develop --release
# Windows, if the maturin launcher is blocked: pwsh scripts/dev_build.ps1
```

## Usage

```python
import numpy as np, neuralforge as nf
nf.cosine_similarity(np.random.rand(768).astype(np.float32),
                     np.random.rand(768).astype(np.float32))
```

See [docs/API_REFERENCE.md](../docs/API_REFERENCE.md). Boundary details:
NumPy buffers are borrowed read-only (zero-copy), the GIL is released around the
Rust compute, and `CoreError` is mapped to `ValueError`.
