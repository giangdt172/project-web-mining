"""Static (matplotlib) renderers for the payloads built in `viz.data`.

Palette: the validated reference categorical order (blue / orange / aqua = tag / question / user)
on the light chart surface. Text always wears ink tokens, never the series colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ROLE_COLOR = {"tag": "#2a78d6", "question": "#eb6834", "user": "#1baf7a", "sink": MUTED}
ROLE_NAME = {"tag": "Tag", "question": "Question", "user": "User", "sink": "Absorbing sink"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f1fc", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)


def _title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10, fontweight="bold", color=INK, pad=8)


def _header(fig: plt.Figure, payload: dict[str, Any], top: float) -> None:
    fig.text(0.015, top, payload["title"], fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.015, top - 0.032, _wrap(payload.get("subtitle", ""), 150), fontsize=8.5, color=INK2, va="top")


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- layered


def render_layered(payload: dict[str, Any], path: Path) -> Path:
    cols = payload["columns"]
    nodes = payload["nodes"]
    by_id = {n["id"]: n for n in nodes}
    per_col = {c: [n for n in nodes if n["col"] == c] for c in range(len(cols))}
    rows = max(len(v) for v in per_col.values())
    height = max(5.2, 0.42 * rows + 2.6)
    fig, ax = plt.subplots(figsize=(13, height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")

    x_of = {c: 0.17 + c * (0.66 / (len(cols) - 1)) for c in range(len(cols))}
    y_of: dict[str, float] = {}
    for c, members in per_col.items():
        n_c = len(members)
        for n in members:
            y_of[n["id"]] = 0.5 if n_c == 1 else 0.90 - 0.80 * n["order"] / (n_c - 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for c, col in enumerate(cols):
        ax.text(x_of[c], 0.985, col["label"].upper(), ha="center", va="bottom", fontsize=8, color=MUTED, fontweight="bold")

    n_qu = sum(1 for e in payload["edges"] if e["layer"] == "question->user" and e["kind"] == "walk")
    sparse_graph = n_qu <= 6  # numeric edge labels only where they stay legible; otherwise hover (HTML)
    # edges first (under nodes)
    for e in payload["edges"]:
        s, t = by_id[e["s"]], by_id[e["t"]]
        x0, y0, x1, y1 = x_of[s["col"]], y_of[s["id"]], x_of[t["col"]], y_of[t["id"]]
        mid = (x0 + x1) / 2
        patch = PathPatch(
            MplPath([(x0, y0), (mid, y0), (mid, y1), (x1, y1)], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]),
            facecolor="none", edgecolor=MUTED if e["kind"] == "walk" else AXIS,
            lw=0.7 + (4.8 if e["kind"] == "walk" else 2.6) * e["wn"], alpha=0.55 if e["kind"] == "walk" else 0.9,
            linestyle="-" if e["kind"] == "walk" else (0, (3, 3)), capstyle="round", zorder=1,
        )
        ax.add_patch(patch)
        if sparse_graph and e["kind"] == "walk" and e["layer"] == "question->user":
            ax.text(
                x0 + (x1 - x0) * 0.78, y0 + (y1 - y0) * 0.78 + 0.008, f"{e['w']:.2f}", fontsize=6.5, color=INK2,
                ha="center", va="bottom", zorder=3,
                bbox={"boxstyle": "round,pad=0.12", "fc": SURFACE, "ec": "none", "alpha": 0.85},
            )
        if sparse_graph and e["kind"] == "leak" and e["w"] >= 0.05:
            ax.text(
                x0 + (x1 - x0) * 0.62, y0 + (y1 - y0) * 0.62, f"leak {e['w']:.2f}", fontsize=6.5, color=MUTED,
                ha="center", va="center", zorder=3,
                bbox={"boxstyle": "round,pad=0.12", "fc": SURFACE, "ec": "none", "alpha": 0.9},
            )

    for n in nodes:
        r = 55 + 330 * n["size"]
        ax.scatter(
            [x_of[n["col"]]], [y_of[n["id"]]], s=r, color=ROLE_COLOR[n["kind"]], edgecolor=SURFACE, linewidth=1.6,
            zorder=4, marker="o" if n["kind"] != "sink" else "X",
        )
        left = n["col"] == 0
        dx = 0.012 + np.sqrt(r) / 2400
        ax.text(
            x_of[n["col"]] + (-dx if left else dx), y_of[n["id"]],
            n["label"], fontsize=8, color=INK, ha="right" if left else "left", va="center", zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.8},
        )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROLE_COLOR[k], markersize=8, label=ROLE_NAME[k])
        for k in ("tag", "question", "user")
    ]
    if "sink" in {n["kind"] for n in nodes}:
        handles.append(Line2D([0], [0], color=AXIS, linestyle=(0, (3, 3)), lw=1.6, label="Absorbed (leak)"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, labelcolor=INK2, ncol=len(handles), bbox_to_anchor=(0.0, -0.02))
    note = "  ".join(payload.get("notes", [])[:1])
    if not sparse_graph:
        note += "  Numeric edge labels are omitted here for legibility; the interactive HTML shows every value on hover."
    ax.text(0.0, -0.075, _wrap(note, 170), fontsize=7.5, color=MUTED, va="top", transform=ax.transAxes)
    _header(fig, payload, 0.985)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10)
    return _save(fig, path)


# ---------------------------------------------------------------------------- charts


def _draw_stacked(ax: plt.Axes, panel: dict[str, Any]) -> None:
    d = panel["data"]
    x = np.asarray(d["x"])
    bottom = np.zeros(len(x))
    for s in d["series"]:
        v = np.asarray(s["values"])
        ax.bar(x, v, bottom=bottom, width=0.78, color=ROLE_COLOR[s["role"]], edgecolor=SURFACE, linewidth=1.2, label=s["name"])
        bottom += v
    _style(ax)
    ax.set_xticks(x)
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3, loc="upper right")
    _title(ax, panel["title"])


def _draw_lines(ax: plt.Axes, panel: dict[str, Any]) -> None:
    d = panel["data"]
    x = np.asarray(d["x"])
    for s in d["series"]:
        color = INK if s["ink"] == "primary" else MUTED
        ax.plot(x, s["values"], color=color, lw=2, ls="--" if s["dash"] else "-", marker="o", ms=4, mec=SURFACE, mew=1.2, label=s["name"])
    _style(ax)
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    _title(ax, panel["title"])


def _draw_dumbbell(ax: plt.Axes, panel: dict[str, Any]) -> None:
    d = panel["data"]
    rows = d["rows"]
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows, strict=True):
        if r["b"] is not None:
            ax.plot([r["a"], r["b"]], [yi, yi], color=AXIS, lw=2, zorder=1, solid_capstyle="round")
            ax.scatter([r["b"]], [yi], s=46, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.6, zorder=2)
        ax.scatter([r["a"]], [yi], s=52, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8, color=INK2)
    ax.set_xlim(0, max(max(r["a"], r["b"] or 0.0) for r in rows) * 1.08)
    ax.set_xlabel(d["xlabel"])
    from matplotlib.lines import Line2D

    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markersize=7, label=d["a_name"])]
    if d["b_name"]:
        h.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=7, label=d["b_name"]))
    ax.legend(handles=h, frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    _title(ax, panel["title"])


def _draw_hist(ax: plt.Axes, panel: dict[str, Any]) -> None:
    """Bins are drawn as equal-width bands, so log-spaced bins read on a log axis (as in the HTML)."""
    d = panel["data"]
    bins = d["bins"]
    cnt = np.array([b["count"] for b in bins])
    dp = d.get("decimals")
    ax.bar(np.arange(len(bins)), cnt, width=0.86, color=ROLE_COLOR[panel["role"]], edgecolor=SURFACE, linewidth=1.0)
    _style(ax)
    step = 1 if len(bins) < 9 else 2
    ax.set_xticks(np.arange(0, len(bins), step))
    ax.set_xticklabels(
        [f"{bins[i]['lo']:.{dp}f}" if dp is not None else f"{bins[i]['lo']:g}" for i in range(0, len(bins), step)],
        rotation=40, ha="right",
    )
    ax.set_xlabel(d["xlabel"] + (" (log bins)" if d.get("xlog") else ""))
    ax.set_ylabel(d["ylabel"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    _title(ax, panel["title"])


def _draw_heatmap(ax: plt.Axes, panel: dict[str, Any], fig: plt.Figure) -> None:
    d = panel["data"]
    v = np.asarray(d["values"])
    im = ax.imshow(np.sqrt(v), cmap=SEQ, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(d["cols"])))
    ax.set_xticklabels(d["cols"], rotation=90, fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(d["rows"])))
    ax.set_yticklabels(d["rows"], fontsize=6.5, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ticks = np.sqrt(np.array([0.0, v.max() * 0.25, v.max() * 0.5, v.max()]))
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t**2:.3f}" for t in ticks])
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    cb.outline.set_visible(False)
    cb.set_label("probability (colour ∝ √p)", fontsize=7.5, color=INK2)
    _title(ax, panel["title"])


def _draw(ax: plt.Axes, panel: dict[str, Any], fig: plt.Figure) -> None:
    kind = panel["type"]
    if kind == "stacked":
        _draw_stacked(ax, panel)
    elif kind == "lines":
        _draw_lines(ax, panel)
    elif kind == "dumbbell":
        _draw_dumbbell(ax, panel)
    elif kind == "hist":
        _draw_hist(ax, panel)
    elif kind == "heatmap":
        _draw_heatmap(ax, panel, fig)
    else:
        raise ValueError(f"unknown panel type {kind!r}")


def render_ppr(payload: dict[str, Any], path: Path) -> Path:
    panels = payload["panels"]
    fig = plt.figure(figsize=(13, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.15], hspace=0.42, wspace=0.22)
    _draw(fig.add_subplot(gs[0, 0]), panels[0], fig)
    _draw(fig.add_subplot(gs[0, 1]), panels[1], fig)
    _draw(fig.add_subplot(gs[1, :]), panels[2], fig)
    _header(fig, payload, 0.985)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.07)
    return _save(fig, path)


def render_overview(payload: dict[str, Any], path: Path) -> Path:
    panels = payload["panels"]
    fig = plt.figure(figsize=(13, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.5, wspace=0.28)
    for i in range(3):
        _draw(fig.add_subplot(gs[0, i]), panels[i], fig)
    _draw(fig.add_subplot(gs[1, :2]), panels[3], fig)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    y = 0.95
    for tile in payload["tiles"]:
        ax.text(0.0, y, tile["value"], fontsize=17, color=INK, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.0, y - 0.075, tile["label"], fontsize=8.5, color=INK2, va="top", transform=ax.transAxes)
        y -= 0.165
    _header(fig, payload, 0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.15)
    return _save(fig, path)
