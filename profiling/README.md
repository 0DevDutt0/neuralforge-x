# profiling — CPU & GPU profiling lab (Phase 6)

Repeatable profiling of the hot paths, split into an **analysis** half (portable,
no special privileges) and a **capture** half (external profilers). Capture
artifacts land in `profiling/out/` (git-ignored); the committed outputs are the
report ([CPU_PROFILE.md](CPU_PROFILE.md)) and chart (`docs/assets/profile_cpu.svg`).

## Layout

| Path | Role |
|------|------|
| `src/main.rs` (`neuralforge_profile`) | sustained-load CPU target driving the real kernels |
| `gpu_workload.py` | sustained-load GPU target driving the `neuralforge_cuda` kernels |
| `analyze.py` | criterion medians → `CPU_PROFILE.md` + `docs/assets/profile_cpu.svg` |
| `scripts/cpu_flamegraph.ps1` | `cargo flamegraph` capture of the CPU target |
| `scripts/gpu_nsight.ps1` | Nsight Systems/Compute capture of the GPU target |

## Analysis (portable)

```bash
cargo bench -p neuralforge_core      # produces target/criterion/**/estimates.json
python profiling/analyze.py          # -> CPU_PROFILE.md + profile_cpu.svg
```

The report turns criterion medians into the scalar-vs-SIMD effectiveness table, a
roofline read (the AVX2+FMA kernel saturates ~33 Gelem/s — load-bound past
384-dim), and an optimization log. See [CPU_PROFILE.md](CPU_PROFILE.md).

## Capture (external profilers)

```bash
# CPU flamegraph of the real kernels under load
pwsh profiling/scripts/cpu_flamegraph.ps1 -Workload topk -Seconds 15

# GPU CUDA timeline + per-kernel metrics
pwsh profiling/scripts/gpu_nsight.ps1 [-Ncu]
```

The `neuralforge_profile` binary builds with the `profiling` cargo profile
(release optimization + line-table debug info) so samples map back to source.

### Environment notes (honest)

- **CPU flamegraph needs elevation on Windows.** `cargo flamegraph` samples via
  ETW (`blondie`), which requires an **administrator** shell; from a normal shell
  it returns `NotAnAdmin`. The criterion analysis path above needs no privileges
  and is the portable default. Linux/macOS (perf/dtrace) need no elevation.
- **GPU capture on Blackwell (`sm_120`) needs a current Nsight.** The scripts run
  and produce a `.nsys-rep`, but the locally-installed Nsight Systems 2024.5.1
  records no CUDA rows for this `sm_120` + CUDA 12.9 stack — the same
  toolchain-freshness gap documented for the CUDA engine (Phase 3). On supported
  hardware/toolkits the same scripts yield the kernel timeline and occupancy.

Each capture is meant to drive an optimization report: hypothesis → change →
measured delta, tied back to [PERFORMANCE.md](../docs/PERFORMANCE.md).
