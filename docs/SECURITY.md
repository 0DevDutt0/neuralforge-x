# Security Policy

## Supported versions

NeuralForge-X is pre-1.0; security fixes land on the latest `0.x` minor only.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead email
**devduttshoji123@gmail.com** with:

- a description of the issue and its impact,
- steps (or a proof-of-concept) to reproduce,
- affected version/commit.

You can expect an acknowledgement within 72 hours and a remediation plan once
the report is triaged.

## Security posture & `unsafe`

NeuralForge-X processes only numeric arrays supplied by the caller; it performs
no network or filesystem I/O in its core paths and has no third-party service
dependencies.

`unsafe` Rust is confined to the SIMD kernels in
[`rust_core/src/simd.rs`](https://github.com/0DevDutt0/neuralforge-x/blob/main/rust_core/src/simd.rs). The invariants are:

- Vectorised functions are annotated `#[target_feature(enable = "avx2,fma")]`
  and are reached **only** after `is_x86_feature_detected!` confirms the
  features at runtime; otherwise a safe scalar path is used.
- Raw-pointer loads operate strictly within validated slice bounds (lengths are
  checked by the safe public wrappers before any kernel is called).
- Equivalence between the `unsafe` SIMD path and the safe scalar reference is
  asserted by `proptest` property tests in CI.

The FFI boundary returns typed errors rather than unwinding across it; panics in
Rust are converted to Python exceptions by PyO3.

## Supply chain

- Dependencies are pinned via a committed `Cargo.lock`.
- CI runs `cargo clippy -D warnings`, `cargo fmt --check`, `ruff`, and `mypy`.
- Adding a dependency requires a justification in the pull request.
