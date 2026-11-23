#!/usr/bin/env python3
"""Visualize the cached heterogeneous transition graph (static PNG + interactive HTML).

Outputs (under `<reports_dir>/figures/graphs/`):
  worked_example.{png,html}          Section 5.1 worked example with the absorbing κ leak
  ego_<tag>_<month>.{png,html}       Tag → Question → User → Tag ego-network (top PPR users)
  ppr_<month>.{png,html}             PPR mass per step, weighted vs unweighted, top-user dumbbell
  overview_<month>.{png,html}        Degree / leak histograms and one-lap tag→tag heatmap
  explorer_<month>.html              FULLY INTERACTIVE explorer: pick seed tags, sliders, live PPR in the browser
  index.html                         Links to everything above
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np

from so_expert_rec.config import load_config
from so_expert_rec.viz import (
    ego_payload,
    load_context,
    overview_payload,
    ppr_payload,
    render_html,
    render_layered,
    render_overview,
    render_ppr,
    worked_example_payload,
)
from so_expert_rec.viz.interactive import explorer_payload, render_explorer

ALL = ("worked", "ego", "ppr", "overview", "explorer")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def _write_index(out: Path, month: str) -> None:
    """List every visualization currently in `out` (not only the ones produced by this run)."""
    titles = {
        "explorer": "Interactive explorer (live PPR in the browser)",
        "worked_example": "Worked example (§5.1)",
        "ego": "Tag ego-network",
        "ppr": "PPR propagation",
        "overview": "Graph structure overview",
    }
    groups: dict[str, dict[str, str]] = {}
    for f in sorted(out.iterdir()):
        if f.suffix not in (".png", ".html") or f.name == "index.html":
            continue
        groups.setdefault(f.stem, {})[f.suffix[1:]] = f.name
    items = ""
    for stem, links in groups.items():
        kind = next((k for k in titles if stem.startswith(k)), None)
        label = f"{titles.get(kind, stem)} — {stem}" if kind in ("ego", "ppr", "overview") else titles.get(kind, stem)
        anchors = " · ".join(
            f'<a href="{html.escape(name)}">{ext.upper()}</a>' for ext, name in sorted(links.items(), reverse=True)
        )
        items += f"<li><b>{html.escape(label)}</b> — {anchors}</li>"
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Graph visualizations</title><body style='font:15px/1.6 system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 16px'>"
        f"<h1>Graph visualizations · {html.escape(month)}</h1><ul>{items}</ul></body>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("reports/sample_ds/tables/tuned_milestone4_config.yaml"))
    parser.add_argument("--month", default="2021-12", help="Snapshot month YYYY-MM (graph must already be cached).")
    parser.add_argument("--tag", action="append", help="Ego-network seed tag (repeatable). Default: most frequent tag.")
    parser.add_argument("--ppr-tags", help="Comma-separated query tags for the PPR page. Default: top-2 tags.")
    parser.add_argument("--top-users", type=int, default=8, help="Users shown in the ego network.")
    parser.add_argument("--top-questions", type=int, default=10, help="Questions shown in the ego network.")
    parser.add_argument("--tags-per-user", type=int, default=2, help="Top tags drawn per user in the ego network.")
    parser.add_argument("--damping", type=float, default=None, help="PPR damping d (default: baselines.b4_damping).")
    parser.add_argument("--only", nargs="+", choices=ALL, default=list(ALL), help="Subset of visualizations.")
    parser.add_argument("--no-static", action="store_true", help="Skip PNG output.")
    parser.add_argument("--no-html", action="store_true", help="Skip interactive HTML output.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <reports_dir>/figures/graphs).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.out or Path(cfg.paths.reports_dir) / "figures" / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str, Path]] = []

    def emit(name: str, label: str, payload: dict, static_fn) -> None:
        if not args.no_static:
            written.append((label, "png", static_fn(payload, out / f"{name}.png")))
        if not args.no_html:
            written.append((label, "html", render_html(payload, out / f"{name}.html")))

    if "worked" in args.only:
        payload, _ = worked_example_payload()
        emit("worked_example", "Worked example (§5.1)", payload, render_layered)

    needs_ctx = any(k in args.only for k in ("ego", "ppr", "overview", "explorer"))
    if needs_ctx:
        ctx = load_context(cfg, args.month, damping=args.damping)
        df = np.diff(ctx.graph.p_tag_to_question.indptr)
        popular = [ctx.tags[int(i)] for i in np.argsort(-df)]
        if "ego" in args.only:
            for tag in args.tag or popular[:1]:
                payload = ego_payload(
                    ctx, tag, top_users=args.top_users, top_questions=args.top_questions,
                    tags_per_user=args.tags_per_user,
                )
                emit(f"ego_{_slug(tag)}_{args.month}", f"Ego network: {tag}", payload, render_layered)
        if "ppr" in args.only:
            tags = [t.strip() for t in args.ppr_tags.split(",")] if args.ppr_tags else popular[:2]
            emit(f"ppr_{args.month}", f"PPR propagation: {', '.join(tags)}", ppr_payload(ctx, tags), render_ppr)
        if "overview" in args.only:
            emit(f"overview_{args.month}", "Graph structure overview", overview_payload(ctx), render_overview)

        if "explorer" in args.only and not args.no_html:
            payload = explorer_payload(ctx, default_tags=args.tag or popular[:1], popular=popular)
            written.append(("Interactive explorer (live PPR)", "html", render_explorer(payload, out / f"explorer_{args.month}.html")))

    _write_index(out, args.month)
    for _, ext, path in written:
        print(f"  [{ext}] {path}")
    print(f"Index: {out / 'index.html'}")


if __name__ == "__main__":
    main()
