# Animated assets

These are **hand-authored animated SVGs** — they animate natively on GitHub via
`<img>` (SMIL/CSS, no JavaScript, no external tooling), so the README is visually
complete with zero recorded media to host:

| File | Shows |
|------|-------|
| [`quickstart.svg`](quickstart.svg) | a typed install + a 5-NN cosine search returning in 3.4 ms |
| [`../hero.svg`](../hero.svg) | the animated hero banner (neural graph, SIMD lanes, GPU ring) |
| [`../vector_search.svg`](../vector_search.svg) | query → SIMD scan → per-thread heaps → merged top-k |
| [`../bench_race.svg`](../bench_race.svg) | the batch-cosine speedup race across backends |
| [`../gpu_pipeline.svg`](../gpu_pipeline.svg) | host⇄device transfer, `sm_120` kernels, intensity routing |
| [`../obs_dashboard.svg`](../obs_dashboard.svg) | live request-rate / p95-latency / readiness panels |

A reproducible **VHS** script ([`scripts/demo.tape`](../../../scripts/demo.tape))
can additionally record a real terminal GIF (`vhs scripts/demo.tape`) for anyone
who prefers a captured session.
