"""SVG chart generation from benchmark results.

Charts are emitted as hand-rolled SVG in the repository's house style (dark navy
panel, Segoe-UI type, one accent colour per backend) so they render natively on
GitHub, diff as text, and need no plotting dependency. The renderer is a generic
grouped-bar builder; the public helpers turn a workload record into the specific
charts referenced by the README and docs.

All inputs are plain dicts (the JSON a run produces), so charts can be
regenerated from a committed results file without re-running the benchmarks.
"""

from __future__ import annotations

import html
from typing import Any, Callable

# One stable accent per backend, used across every chart.
PALETTE: dict[str, str] = {
    "python": "#f59e0b",
    "numpy": "#38bdf8",
    "rust": "#34d399",
    "cuda": "#a78bfa",
    "triton": "#f472b6",
    "torch": "#fb7185",
    "hnsw": "#fbbf24",
    # Profiling-lab series (scalar vs SIMD).
    "scalar": "#64748b",
    "simd": "#34d399",
}
_DEFAULT_COLOR = "#94a3b8"

_W, _H = 860, 440
_LEFT, _RIGHT, _TOP, _BOTTOM = 92, 824, 92, 366


def _nice_ceil(value: float) -> float:
    """Rounds ``value`` up to a visually pleasant axis maximum."""
    if value <= 0:
        return 1.0
    import math

    exp = math.floor(math.log10(value))
    base = 10.0**exp
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if value <= step * base:
            return step * base
    return 10.0 * base


def _color(series: str) -> str:
    return PALETTE.get(series, _DEFAULT_COLOR)


def grouped_bar_svg(
    *,
    title: str,
    subtitle: str,
    group_labels: list[str],
    series: list[str],
    value_at: Callable[[int, str], float | None],
    value_label: Callable[[float], str],
    y_max: float | None = None,
    baseline: tuple[float, str] | None = None,
) -> str:
    """Renders a grouped vertical-bar chart as a standalone SVG string.

    Args:
        title / subtitle: header text.
        group_labels: x-axis groups (e.g. corpus sizes).
        series: bar series within each group (backend names), drives colour/order.
        value_at: ``(group_index, series) -> value | None``; ``None`` omits a bar.
        value_label: formats a value for the label above its bar.
        y_max: optional fixed axis maximum; otherwise derived from the data.
        baseline: optional ``(value, label)`` dashed reference line.
    """
    values = [[value_at(g, s) for s in series] for g in range(len(group_labels))]
    flat = [v for row in values for v in row if v is not None]
    top = y_max if y_max is not None else _nice_ceil(max(flat) if flat else 1.0)
    plot_h = _BOTTOM - _TOP
    plot_w = _RIGHT - _LEFT

    def y_of(v: float) -> float:
        return _BOTTOM - (v / top) * plot_h

    parts: list[str] = [_panel(title, subtitle)]

    # Y grid + labels (5 steps).
    parts.append('<g class="ax">')
    for i in range(6):
        v = top * i / 5
        y = y_of(v)
        parts.append(
            f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_RIGHT}" y2="{y:.1f}" '
            f'stroke="#1b2944" stroke-width="1"/>'
            f'<text x="{_LEFT - 12}" y="{y + 4:.1f}" text-anchor="end">{value_label(v)}</text>'
        )
    parts.append(
        f'<line x1="{_LEFT}" y1="{_BOTTOM}" x2="{_RIGHT}" y2="{_BOTTOM}" '
        f'stroke="#2a3a59" stroke-width="1.5"/></g>'
    )

    # Bars.
    n_groups = len(group_labels)
    group_w = plot_w / n_groups
    bar_gap = 8.0
    inner = group_w * 0.74
    bw = (inner - bar_gap * (len(series) - 1)) / max(1, len(series))
    for g, label in enumerate(group_labels):
        gx = _LEFT + group_w * g + (group_w - inner) / 2
        for si, s in enumerate(series):
            v = values[g][si]
            x = gx + si * (bw + bar_gap)
            if v is None:
                continue
            y = y_of(v)
            h = _BOTTOM - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                f'rx="4" fill="{_color(s)}"/>'
            )
            parts.append(
                f'<text x="{x + bw / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" '
                f'class="v">{html.escape(value_label(v))}</text>'
            )
        parts.append(
            f'<text x="{_LEFT + group_w * g + group_w / 2:.1f}" y="{_BOTTOM + 24:.0f}" '
            f'text-anchor="middle" class="m">{html.escape(label)}</text>'
        )

    if baseline is not None:
        bval, blabel = baseline
        if bval <= top:
            by = y_of(bval)
            parts.append(
                f'<line x1="{_LEFT}" y1="{by:.1f}" x2="{_RIGHT}" y2="{by:.1f}" '
                f'stroke="#f59e0b" stroke-width="2" stroke-dasharray="7 5"/>'
                f'<text x="{_RIGHT - 8}" y="{by - 7:.1f}" text-anchor="end" class="m" '
                f'fill="#f59e0b">{html.escape(blabel)}</text>'
            )

    parts.append(_legend(series))
    return _document("".join(parts), title)


_FONT = "font-family:Segoe UI,system-ui,Arial,sans-serif"


def _panel(title: str, subtitle: str) -> str:
    return (
        f'<rect width="{_W}" height="{_H}" rx="14" fill="url(#cbg)"/>'
        f'<text x="40" y="42" class="t" font-size="20" font-weight="800">'
        f"{html.escape(title)}</text>"
        f'<text x="40" y="66" class="m">{html.escape(subtitle)}</text>'
    )


def _legend(series: list[str]) -> str:
    parts = ['<g class="m">']
    x = _LEFT
    y = _H - 22
    for s in series:
        parts.append(
            f'<rect x="{x}" y="{y - 10}" width="13" height="13" rx="3" fill="{_color(s)}"/>'
        )
        parts.append(f'<text x="{x + 19}" y="{y + 1}">{html.escape(s)}</text>')
        x += 30 + 9 * len(s)
    parts.append("</g>")
    return "".join(parts)


def _document(body: str, aria: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'width="{_W}" height="{_H}" role="img" aria-label="{html.escape(aria)}">'
        '<defs><linearGradient id="cbg" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#0b1020"/><stop offset="1" stop-color="#0e1830"/>'
        "</linearGradient></defs>"
        "<style>"
        f".t{{{_FONT};fill:#e6edf3}}"
        f".m{{{_FONT};fill:#9fb0c9;font-size:13px}}"
        f".v{{{_FONT};fill:#e6edf3;font-size:12px;font-weight:700}}"
        f".ax{{{_FONT};fill:#6c7a96;font-size:12px}}"
        "</style>"
        f"{body}</svg>\n"
    )


# --------------------------------------------------------------------------
# Workload → chart helpers.
# --------------------------------------------------------------------------
def _measurements(size: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["backend"]: m for m in size["measurements"]}


def _series_order(workload: dict[str, Any]) -> list[str]:
    """Backends present anywhere in the workload, in a stable, sensible order."""
    seen: list[str] = []
    for size in workload["sizes"]:
        for m in size["measurements"]:
            if m["backend"] not in seen:
                seen.append(m["backend"])
    preferred = ["python", "numpy", "rust", "cuda", "triton", "torch", "hnsw"]
    return sorted(seen, key=lambda b: preferred.index(b) if b in preferred else len(preferred))


def _group_label(workload: dict[str, Any], size: dict[str, Any]) -> str:
    params = size["params"]
    if "dim" in params and "corpus" not in params:
        return f"d={params['dim']}"
    corpus = params.get("corpus", "")
    return f"{corpus:,}" if isinstance(corpus, int) else str(corpus)


def _speedup_chart(
    workload: dict[str, Any], *, baseline: str, series: list[str], title: str, subtitle: str
) -> str:
    """A 'speedup vs `baseline`' chart over an explicit series subset."""
    groups = [_group_label(workload, s) for s in workload["sizes"]]

    def value_at(g: int, s: str) -> float | None:
        ms = _measurements(workload["sizes"][g])
        if s not in ms or baseline not in ms:
            return None
        base = ms[baseline]["best_ms"]
        return round(base / ms[s]["best_ms"], 2) if ms[s]["best_ms"] > 0 else None

    return grouped_bar_svg(
        title=title,
        subtitle=subtitle,
        group_labels=groups,
        series=series,
        value_at=value_at,
        value_label=lambda v: f"{v:g}×",
        baseline=(1.0, f"1× — {baseline} baseline"),
    )


def _is_approximate(workload: dict[str, Any], backend: str) -> bool:
    """Whether a backend reports approximate results (e.g. the ANN index)."""
    return any(
        m["backend"] == backend and m.get("approximate")
        for size in workload["sizes"]
        for m in size["measurements"]
    )


def speedup_chart(workload: dict[str, Any], *, subtitle: str) -> str:
    """A 'speedup vs baseline' chart (higher is better) for a workload.

    Approximate backends (the HNSW index) are excluded: their speedup is on a
    different order of magnitude and a different correctness basis, so mixing them
    into an exact-scan comparison would both mislead and wreck the axis scale.
    """
    baseline = workload["baseline"]
    series = [
        s for s in _series_order(workload) if s != baseline and not _is_approximate(workload, s)
    ]
    return _speedup_chart(
        workload,
        baseline=baseline,
        series=series,
        title=f"{workload['workload']} — speedup vs {baseline}",
        subtitle=subtitle,
    )


def memory_chart(workload: dict[str, Any], *, subtitle: str) -> str:
    """Peak host allocation per backend (MiB, lower is better)."""
    series = _series_order(workload)
    groups = [_group_label(workload, s) for s in workload["sizes"]]

    def value_at(g: int, s: str) -> float | None:
        ms = _measurements(workload["sizes"][g])
        if s not in ms:
            return None
        return round(ms[s]["peak_host_kib"] / 1024.0, 2)

    return grouped_bar_svg(
        title=f"{workload['workload']} — peak host memory",
        subtitle=subtitle,
        group_labels=groups,
        series=series,
        value_at=value_at,
        value_label=lambda v: f"{v:g}",
    )


def find_workload(doc: dict[str, Any], name: str) -> dict[str, Any] | None:
    for w in doc.get("workloads", []):
        if w["workload"] == name:
            return w
    return None


def render_charts(doc: dict[str, Any]) -> dict[str, str]:
    """Builds the standard chart set from a results document.

    Returns a mapping of ``filename -> svg``. Missing workloads are skipped, so
    a CPU-only run still produces the charts it can.
    """
    env = doc.get("environment", {})
    cpu = env.get("processor", "CPU")
    if "Intel64" in cpu or "GenuineIntel" in cpu:
        cpu = "Intel Core Ultra 9 (AVX2)"
    repeats = doc.get("config", {}).get("repeats", "?")
    base_sub = f"best-of-{repeats} · {cpu}"
    charts: dict[str, str] = {}

    pairwise = find_workload(doc, "pairwise_cosine")
    if pairwise:
        charts["bench_pairwise.svg"] = speedup_chart(
            pairwise,
            subtitle=f"single vector pair, dimension sweep · {base_sub}. Higher is better.",
        )

    batch = find_workload(doc, "batch_cosine")
    if batch:
        q = doc.get("config", {}).get("queries", "?")
        charts["bench_batch.svg"] = speedup_chart(
            batch, subtitle=f"{q} queries × corpus, 768-dim · {base_sub}. Higher is better."
        )
        charts["bench_memory.svg"] = memory_chart(
            batch, subtitle=f"transient Python-heap peak (MiB) · {base_sub}. Lower is better."
        )
        gpu_series = [s for s in _series_order(batch) if s in ("cuda", "triton", "torch")]
        if gpu_series:
            charts["bench_gpu.svg"] = _speedup_chart(
                batch,
                baseline="rust",
                series=gpu_series,
                title="batch_cosine — GPU speedup vs Rust (CPU)",
                subtitle=f"end-to-end, host-to-host · {base_sub}. Higher is better.",
            )

    topk = find_workload(doc, "topk_search")
    if topk:
        charts["bench_topk.svg"] = speedup_chart(
            topk,
            subtitle=f"exact k-NN scan; single query · {base_sub}. Higher is better.",
        )

    return charts
