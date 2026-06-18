"""Command-line interface: ``python -m benchmark_lab``.

Subcommands:

* ``run``    — execute the workloads and write a results JSON.
* ``charts`` — render SVG charts from a results JSON into ``docs/assets/``.
* ``report`` — render a Markdown summary from a results JSON.
* ``all``    — ``run`` then ``charts`` then ``report`` in one shot.

Examples::

    python -m benchmark_lab run --quick
    python -m benchmark_lab all
    python -m benchmark_lab charts --results benchmark_lab/results/cross_stack.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .charts import render_charts
from .report import render_markdown
from .workloads import BenchConfig, run_all

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS = _ROOT / "benchmark_lab" / "results" / "cross_stack.json"
_DEFAULT_ASSETS = _ROOT / "docs" / "assets"


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible, else the path itself."""
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"results file not found: {path}\nRun `python -m benchmark_lab run` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = BenchConfig.quick() if args.quick else BenchConfig(use_gpu=not args.no_gpu)
    if args.repeats:
        cfg = _with_repeats(cfg, args.repeats)
    print(f"Running benchmarks (quick={args.quick}, gpu={cfg.use_gpu}, repeats={cfg.repeats}) ...")
    doc = run_all(cfg)
    out = Path(args.out) if args.out else _DEFAULT_RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {_rel(out)}")
    return doc


def _with_repeats(cfg: BenchConfig, repeats: int) -> BenchConfig:
    from dataclasses import replace

    return replace(cfg, repeats=repeats)


def _cmd_charts(args: argparse.Namespace, doc: dict[str, Any] | None = None) -> None:
    doc = doc or _load(Path(args.results) if args.results else _DEFAULT_RESULTS)
    out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_ASSETS
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = render_charts(doc)
    for name, svg in charts.items():
        (out_dir / name).write_text(svg, encoding="utf-8")
    print(f"Wrote {len(charts)} charts to {_rel(out_dir)}: {', '.join(charts)}")


def _cmd_report(args: argparse.Namespace, doc: dict[str, Any] | None = None) -> None:
    doc = doc or _load(Path(args.results) if args.results else _DEFAULT_RESULTS)
    md = render_markdown(doc)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)


def _cmd_all(args: argparse.Namespace) -> None:
    doc = _cmd_run(args)
    _cmd_charts(args, doc)
    # Print the Markdown summary to stdout (here `--out` is the results path).
    print()
    print(render_markdown(doc))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmark_lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run benchmarks, write results JSON")
    run_p.add_argument("--quick", action="store_true", help="tiny/fast config (CPU only)")
    run_p.add_argument("--no-gpu", action="store_true", help="skip GPU backends")
    run_p.add_argument("--repeats", type=int, default=0, help="override timed iterations")
    run_p.add_argument("--out", help="results JSON path")
    run_p.set_defaults(func=lambda a: _cmd_run(a))

    charts_p = sub.add_parser("charts", help="render SVG charts from results JSON")
    charts_p.add_argument("--results", help="results JSON path")
    charts_p.add_argument("--out-dir", help="output directory for SVGs")
    charts_p.set_defaults(func=_cmd_charts)

    report_p = sub.add_parser("report", help="render a Markdown summary from results JSON")
    report_p.add_argument("--results", help="results JSON path")
    report_p.add_argument("--out", help="write Markdown here instead of stdout")
    report_p.set_defaults(func=_cmd_report)

    all_p = sub.add_parser("all", help="run, then charts, then report")
    all_p.add_argument("--quick", action="store_true", help="tiny/fast config (CPU only)")
    all_p.add_argument("--no-gpu", action="store_true", help="skip GPU backends")
    all_p.add_argument("--repeats", type=int, default=0, help="override timed iterations")
    all_p.add_argument("--out", help="results JSON path")
    all_p.add_argument("--results", help=argparse.SUPPRESS)
    all_p.add_argument("--out-dir", help="output directory for SVGs")
    all_p.set_defaults(func=_cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
