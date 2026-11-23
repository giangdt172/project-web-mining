"""Standalone interactive HTML renderers (vanilla SVG + JS, no network or CDN needed)."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

_DIR = Path(__file__).parent


def _as_page(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a bare layered payload as a one-panel page."""
    if payload.get("type") == "page":
        return payload
    if payload.get("type") == "layered":
        return {
            "type": "page",
            "title": payload["title"],
            "subtitle": payload.get("subtitle", ""),
            "panels": [{"type": "layered", "title": "", "note": "", "data": payload}],
            "notes": payload.get("notes", []),
        }
    raise ValueError(f"unsupported payload type {payload.get('type')!r}")


def _assemble(template: str, title: str, payload: dict[str, Any]) -> str:
    # `</` would terminate the inline <script>; `<\/` is identical once parsed as JSON.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    text = (_DIR / template).read_text(encoding="utf-8")
    text = text.replace("/*__KIT_CSS__*/", (_DIR / "_kit.css").read_text(encoding="utf-8"))
    text = text.replace("/*__KIT_JS__*/", (_DIR / "_kit.js").read_text(encoding="utf-8"))
    return text.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", data)


def _write(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def render_html(payload: dict[str, Any], path: Path) -> Path:
    """Write a self-contained hover/inspect page for a layered or multi-panel `payload`."""
    page = _as_page(payload)
    return _write(path, _assemble("template.html", page["title"], page))


# ------------------------------------------------------------------ explorer


def _b64(arr: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _pack_csr(mat: sparse.csr_matrix) -> dict[str, Any]:
    mat = mat.tocsr()
    mat.sort_indices()
    return {"ip": _b64(mat.indptr, "<i4"), "ix": _b64(mat.indices, "<i4"), "dt": _b64(mat.data, "<f4"), "shape": list(mat.shape)}


def explorer_payload(ctx: Any, *, default_tags: list[str], popular: list[str]) -> dict[str, Any]:
    """Pack the full transition graph (weighted and, if cached, unweighted) for in-browser PPR."""

    def pack(g: Any) -> dict[str, Any]:
        return {
            "tq": _pack_csr(g.p_tag_to_question),
            "qu": _pack_csr(g.p_question_to_user),
            "ut": _pack_csr(g.p_user_to_tag),
            "kappa": float(g.kappa),
        }

    graphs = {"weighted": pack(ctx.graph)}
    if ctx.graph_unweighted is not None:
        graphs["unweighted"] = pack(ctx.graph_unweighted)
    return {
        "month": ctx.month,
        "damping": ctx.damping,
        "tags": ctx.tags,
        "userIds": [int(u) for u in ctx.user_ids],
        "qTitles": [ctx.question_title(i) for i in range(ctx.graph.num_questions)],
        "graphs": graphs,
        "defaultTags": default_tags,
        "popular": popular,
    }


def render_explorer(payload: dict[str, Any], path: Path) -> Path:
    """Write the fully interactive graph explorer (PPR is recomputed live in the browser)."""
    title = f"Graph explorer · snapshot {payload['month']}"
    return _write(path, _assemble("explorer.html", title, payload))
