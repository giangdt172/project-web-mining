"""Data extraction for graph visualizations (renderer-agnostic).

Everything here turns the cached sparse `HeterogeneousGraph` into small, JSON-serialisable
payloads that both the matplotlib (`static.py`) and standalone-HTML (`interactive.py`)
renderers consume, so the two outputs always agree.

Layered payload (`kind="layered"`):
    {"columns": [{"key", "label"}], "nodes": [...], "edges": [...], "notes": [...]}
Node: ``id, col, order, kind (tag|question|user|sink), label, size (0..1), tip [[k, v], ...]``
Edge: ``s, t, w (true probability), wn (0..1 width, normalised within the edge layer),
layer (e.g. "tag->question"), kind ("walk"|"leak")``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from so_expert_rec.config import ExperimentConfig, cache_key_for_stage
from so_expert_rec.graph.builder import HeterogeneousGraph, load_heterogeneous_graph
from so_expert_rec.graph.ppr import ppr_batch

SINK_ID = "sink"


# --------------------------------------------------------------------------- context


@dataclass
class GraphContext:
    """Graphs + human-readable labels for one monthly snapshot."""

    month: str
    graph: HeterogeneousGraph
    graph_unweighted: HeterogeneousGraph | None
    tags: list[str]
    user_ids: np.ndarray
    question_ids: np.ndarray
    question_titles: dict[int, str]
    damping: float

    def tag_index(self, name: str) -> int:
        try:
            return self.tags.index(name)
        except ValueError as exc:
            raise KeyError(f"tag {name!r} not in vocabulary") from exc

    def question_title(self, q_idx: int) -> str:
        qid = int(self.question_ids[q_idx])
        return self.question_titles.get(qid, f"question {qid}")


def _load_titles(interim_dir: Path) -> dict[int, str]:
    path = Path(interim_dir) / "questions.parquet"
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["Id", "Title"])
    ids = table.column("Id").to_pylist()
    titles = table.column("Title").to_pylist()
    return {int(i): str(t) for i, t in zip(ids, titles, strict=True) if i is not None and t}


def load_context(config: ExperimentConfig, month: str, damping: float | None = None) -> GraphContext:
    """Load cached weighted + unweighted graphs for `month` and node labels (no recomputation)."""
    from so_expert_rec.features.snapshot import PrecomputedStore

    month_dir = Path(config.paths.snapshots_dir) / month
    w_dir = month_dir / cache_key_for_stage("graph", config)
    if not (w_dir / "graph_meta.json").exists():
        raise FileNotFoundError(
            f"No cached weighted graph at {w_dir}. Run scripts/03_build_snapshots.py with this config first."
        )
    u_key = "graph_unweighted_aa" if config.graph.include_asker_answerer else "graph_unweighted"
    u_dir = month_dir / u_key
    store = PrecomputedStore.load(config.paths.processed_dir)
    return GraphContext(
        month=month,
        graph=load_heterogeneous_graph(w_dir),
        graph_unweighted=load_heterogeneous_graph(u_dir) if (u_dir / "graph_meta.json").exists() else None,
        tags=store.tags,
        user_ids=store.user_ids,
        question_ids=store.question_ids,
        question_titles=_load_titles(config.paths.interim_dir),
        damping=float(config.baselines.b4_damping if damping is None else damping),
    )


# --------------------------------------------------------------------------- helpers


def _shorten(text: str, n: int = 32) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x >= 0.001 else f"{x:.2e}"


def _normalise_layer_widths(edges: list[dict[str, Any]]) -> None:
    """Fill `wn` (0..1) per edge layer so thickness is comparable within a layer."""
    by_layer: dict[str, float] = {}
    for e in edges:
        by_layer[e["layer"]] = max(by_layer.get(e["layer"], 0.0), e["w"])
    for e in edges:
        peak = by_layer[e["layer"]]
        e["wn"] = float(e["w"] / peak) if peak > 0 else 0.0


def _barycentre_order(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], columns: int) -> None:
    """Assign `order` per column. Column 0/2 are seeded by caller order; others by barycentre.

    A couple of left->right sweeps are enough for these small (<= ~40 node) layered graphs.
    """
    by_id = {n["id"]: n for n in nodes}
    cols: dict[int, list[dict[str, Any]]] = {c: [] for c in range(columns)}
    for n in nodes:
        cols[n["col"]].append(n)
    for members in cols.values():
        members.sort(key=lambda n: n.get("seed_rank", 0.0))
        for i, n in enumerate(members):
            n["order"] = float(i)
    for _ in range(2):
        for c in range(1, columns):
            for n in cols[c]:
                if n.get("pinned"):
                    continue
                neigh = [
                    by_id[e["s"]]["order"] / max(1, len(cols[by_id[e["s"]]["col"]]) - 1)
                    for e in edges
                    if e["t"] == n["id"] and e["kind"] == "walk" and by_id[e["s"]]["col"] == c - 1
                ]
                if neigh:
                    n["seed_rank"] = float(np.mean(neigh))
            movable = [n for n in cols[c] if not n.get("pinned")]
            movable.sort(key=lambda n: n.get("seed_rank", 0.0))
            fixed = [n for n in cols[c] if n.get("pinned")]
            for i, n in enumerate(movable + fixed):
                n["order"] = float(i)


# --------------------------------------------------------------------------- layered payload


def build_layered_payload(
    graph: HeterogeneousGraph,
    *,
    tag_labels: list[str],
    question_labels: list[str],
    user_labels: list[str],
    seed_tags: list[int],
    question_idx: list[int],
    user_idx: list[int],
    next_tag_idx: list[int],
    user_scores: np.ndarray | None = None,
    question_tips: dict[int, list[list[str]]] | None = None,
    user_tips: dict[int, list[list[str]]] | None = None,
    title: str = "",
    subtitle: str = "",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the Tag -> Question -> User -> Tag layered payload for an explicit node selection.

    Edge probabilities are read straight from the graph's transition blocks, so they are exact for
    the shown edges. The absorbed mass (kappa leak + unanswered question) of each shown question is
    drawn as dashed edges into a single sink node.
    """
    p_tq = graph.p_tag_to_question.tocsr()
    p_qu = graph.p_question_to_user.tocsr()
    p_ut = graph.p_user_to_tag.tocsr()

    columns = [
        {"key": "seed", "label": "Seed tag(s)"},
        {"key": "question", "label": "Past questions"},
        {"key": "user", "label": "Users (answerers)"},
        {"key": "tag", "label": "Tags reached"},
    ]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    q_rowsum = np.asarray(p_qu.sum(axis=1)).ravel()

    for rank, t in enumerate(seed_tags):
        nodes.append(
            {
                "id": f"s{t}", "col": 0, "kind": "tag", "label": tag_labels[t], "size": 1.0,
                "seed_rank": float(rank), "pinned": True,
                "tip": [["node", "tag (seed)"], ["|Q_t|", f"{p_tq[t].nnz:,} questions"]],
            }
        )
    for q in question_idx:
        mass = float(q_rowsum[q])
        tip = [["node", "question"], ["answered mass", _fmt(mass)], ["absorbed (leak)", _fmt(1.0 - mass)]]
        tip += (question_tips or {}).get(q, [])
        nodes.append(
            {
                "id": f"q{q}", "col": 1, "kind": "question", "label": _shorten(question_labels[q]),
                "size": mass, "tip": tip,
            }
        )
    scores = user_scores if user_scores is not None else np.zeros(len(user_labels))
    top_score = float(max((scores[u] for u in user_idx), default=0.0)) or 1.0
    for rank, u in enumerate(user_idx):
        tip = [["node", "user"], ["PPR score", f"{scores[u]:.4f}"]] if user_scores is not None else [["node", "user"]]
        tip += (user_tips or {}).get(u, [])
        nodes.append(
            {
                "id": f"u{u}", "col": 2, "kind": "user", "label": user_labels[u],
                "size": float(scores[u] / top_score) if user_scores is not None else 0.5,
                "seed_rank": float(rank), "tip": tip,
            }
        )
    for rank, t in enumerate(next_tag_idx):
        nodes.append(
            {
                "id": f"t{t}", "col": 3, "kind": "tag", "label": tag_labels[t], "size": 0.5,
                "seed_rank": float(rank),
                "tip": [["node", "tag (next lap)" if t not in seed_tags else "tag (seed, loop-back)"]],
            }
        )

    # Tag -> Question
    shown_questions = set(question_idx)
    for t in seed_tags:
        row = p_tq.getrow(t)
        for q, w in zip(row.indices, row.data, strict=True):
            if int(q) in shown_questions:
                edges.append({"s": f"s{t}", "t": f"q{int(q)}", "w": float(w), "layer": "tag->question", "kind": "walk"})
    # Question -> User (+ leak into sink)
    shown_users = set(user_idx)
    any_leak = False
    for q in question_idx:
        row = p_qu.getrow(q)
        hidden = 0.0
        for u, w in zip(row.indices, row.data, strict=True):
            if int(u) in shown_users:
                edges.append({"s": f"q{q}", "t": f"u{int(u)}", "w": float(w), "layer": "question->user", "kind": "walk"})
            else:
                hidden += float(w)
        leak = max(0.0, 1.0 - float(q_rowsum[q]))
        if leak > 1e-9:
            any_leak = True
            edges.append({"s": f"q{q}", "t": SINK_ID, "w": leak, "layer": "question->user", "kind": "leak"})
        if hidden > 1e-9:
            for n in nodes:
                if n["id"] == f"q{q}":
                    n["tip"].append(["to users not shown", _fmt(hidden)])
    if any_leak:
        nodes.append(
            {
                "id": SINK_ID, "col": 2, "kind": "sink", "label": "∅ absorbed", "size": 0.5,
                "seed_rank": 1e6, "pinned": True,
                "tip": [["node", "absorbing sink"], ["what", "kappa leak + unanswered questions; never renormalised"]],
            }
        )
    # User -> Tag
    shown_tags = set(next_tag_idx)
    for u in user_idx:
        row = p_ut.getrow(u)
        for t, w in zip(row.indices, row.data, strict=True):
            if int(t) in shown_tags:
                edges.append({"s": f"u{u}", "t": f"t{int(t)}", "w": float(w), "layer": "user->tag", "kind": "walk"})

    _normalise_layer_widths(edges)
    _barycentre_order(nodes, edges, columns=len(columns))
    return {
        "type": "layered", "title": title, "subtitle": subtitle, "columns": columns,
        "nodes": nodes, "edges": edges, "notes": notes or [],
    }


# --------------------------------------------------------------------------- ego network


def _tag_restart(graph: HeterogeneousGraph, seeds: dict[int, float]) -> np.ndarray:
    s = np.zeros(graph.num_nodes, dtype=np.float64)
    total = float(sum(seeds.values()))
    for t, w in seeds.items():
        s[graph.tag_offset + t] = w / total
    return s


def ppr_user_scores(graph: HeterogeneousGraph, seeds: dict[int, float], damping: float) -> np.ndarray:
    """Exact absorbing PPR user scores `x[U]` for a tag-restart distribution."""
    x = ppr_batch(graph.p_transpose, _tag_restart(graph, seeds)[:, None], damping=damping)[:, 0]
    return x[graph.user_slice]


def ego_payload(
    ctx: GraphContext,
    tag: str,
    *,
    top_users: int = 8,
    top_questions: int = 10,
    tags_per_user: int = 2,
) -> dict[str, Any]:
    """Tag ego-network: seed tag -> its most relevant questions -> top-PPR users -> their top tags."""
    g = ctx.graph
    t = ctx.tag_index(tag)
    scores = ppr_user_scores(g, {t: 1.0}, ctx.damping)
    users = [int(u) for u in np.argsort(-scores)[:top_users] if scores[u] > 0]

    p_tq, p_qu, p_ut = g.p_tag_to_question.tocsr(), g.p_question_to_user.tocsr(), g.p_user_to_tag.tocsr()
    q_cand = p_tq.getrow(t).indices
    to_shown = np.asarray(p_qu[q_cand][:, users].sum(axis=1)).ravel() if len(users) else np.zeros(len(q_cand))
    keep = np.argsort(-to_shown)[:top_questions]
    questions = [int(q_cand[i]) for i in keep if to_shown[i] > 0]

    next_tags: list[int] = []
    for u in users:
        row = p_ut.getrow(u)
        for j in np.argsort(-row.data)[:tags_per_user]:
            ti = int(row.indices[j])
            if ti not in next_tags:
                next_tags.append(ti)
    if t not in next_tags:
        next_tags.insert(0, t)

    user_tips = {}
    for u in users:
        row = p_ut.getrow(u)
        top = np.argsort(-row.data)[:3]
        user_tips[u] = [
            ["answered questions (in graph)", f"{int(p_qu[:, u].nnz):,}"],
            ["top tags", ", ".join(f"{ctx.tags[int(row.indices[j])]} ({row.data[j]:.2f})" for j in top)],
        ]
    n_q = p_tq.getrow(t).nnz
    return build_layered_payload(
        g,
        tag_labels=ctx.tags,
        question_labels=[ctx.question_title(i) for i in range(g.num_questions)],
        user_labels=[f"user {int(uid)}" for uid in ctx.user_ids],
        seed_tags=[t],
        question_idx=questions,
        user_idx=users,
        next_tag_idx=next_tags,
        user_scores=scores,
        user_tips=user_tips,
        title=f"Ego network of tag “{tag}” · snapshot {ctx.month}",
        subtitle=(
            f"One lap of the Tag → Question → User → Tag cycle. Showing the top {len(users)} users by absorbing PPR "
            f"(d={ctx.damping:g}) and the {len(questions)} of {n_q:,} questions that send them the most mass."
        ),
        notes=[
            "Edge thickness is proportional to transition probability and normalised within each layer; hover for exact values.",
            (
                f"Tag → Question is uniform 1/|Q_t| (= 1/{n_q:,}). Dashed edges are mass absorbed at a question "
                f"(κ={g.kappa:.3f} leak, or the question has no answers) and are not renormalised."
            ),
            "User → Tag shows each user's top tags only; the remainder of their row is not drawn.",
        ],
    )


# --------------------------------------------------------------------------- worked example


def worked_example_payload() -> tuple[dict[str, Any], HeterogeneousGraph]:
    """Exact Section 5.1 worked example (Q1/Q2 + a lone weak answer Q3) as a layered payload."""
    from so_expert_rec.features.quality import compute_answer_quality
    from so_expert_rec.graph.builder import build_transition_graph

    _, _, q_tau = compute_answer_quality(
        score_m=np.array([15, 7, 8, 0], dtype=np.int32),
        accepted_m=np.array([1, 0, 0, 0], dtype=np.int8),
        answer_age_days=np.array([10.0, 10.0, 5.0, 5.0]),
        epsilon=0.1, beta=1.0, tau_days=None,
    )
    tq = sparse.csr_matrix(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
    qu = sparse.csr_matrix(
        np.array([[q_tau[0], q_tau[1], 0.0], [q_tau[2], 0.0, 0.0], [0.0, 0.0, q_tau[3]]])
    )
    ut = sparse.csr_matrix(
        np.array(
            [
                [(q_tau[0] + q_tau[2]) / 2.0] * 2,
                [q_tau[1] / 2.0] * 2,
                [q_tau[3], 0.0],
            ]
        )
    )
    graph = build_transition_graph(
        tag_question_adj=tq, question_user_qtau=qu, user_tag_esum=ut,
        kappa_multiplier=1.0, kappa_override=1.0,
    )
    q_tips = {
        0: [["answers", f"Alice q={q_tau[0]:.2f} (+15, accepted) · Bob q={q_tau[1]:.2f} (+7)"]],
        1: [["answers", f"Alice q={q_tau[2]:.2f} (+8)"]],
        2: [["answers", f"Charlie q={q_tau[3]:.2f} (lone weak answer)"]],
    }
    payload = build_layered_payload(
        graph,
        tag_labels=["pytorch", "cuda"],
        question_labels=["Q1: CUDA OOM", "Q2: GPU memory leak", "Q3: lone weak answer"],
        user_labels=["Alice", "Bob", "Charlie"],
        seed_tags=[0],
        question_idx=[0, 1, 2],
        user_idx=[0, 1, 2],
        next_tag_idx=[0, 1],
        question_tips=q_tips,
        title="Worked example (§5.1): where each question's mass goes",
        subtitle=(
            "κ = 1, ε = 0.1, β = 1, no decay. Q1 → Alice 0.55, Bob 0.31, leak 0.14; Q2 → Alice 0.70, leak 0.30; "
            "a lone weak answer (q = 0.1) earns only 0.09."
        ),
        notes=[
            "Reproduces the transition table of Section 5.1 using the real build_transition_graph(...).",
            "The κ leak keeps one weak answer from capturing a whole question's mass; leaked mass is absorbed, not renormalised.",
        ],
    )
    return payload, graph


# --------------------------------------------------------------------------- PPR propagation


def ppr_mass_by_step(
    graph: HeterogeneousGraph, seeds: dict[int, float], damping: float, steps: int = 13
) -> dict[str, np.ndarray]:
    """Mass of each Neumann term `(1-d) (d P^T)^k s` split by node type, for k = 0..steps-1."""
    term = (1.0 - damping) * _tag_restart(graph, seeds)
    out = {"tag": [], "question": [], "user": []}
    for _ in range(steps):
        out["tag"].append(float(term[graph.tag_slice].sum()))
        out["question"].append(float(term[graph.question_slice].sum()))
        out["user"].append(float(term[graph.user_slice].sum()))
        term = damping * (graph.p_transpose @ term)
    return {k: np.asarray(v) for k, v in out.items()}


def ppr_payload(ctx: GraphContext, tags: list[str], *, steps: int = 13, top_users: int = 12) -> dict[str, Any]:
    """Panels showing how PPR mass flows around the cycle and which users end up with it."""
    g, d = ctx.graph, ctx.damping
    seeds = {ctx.tag_index(t): 1.0 for t in tags}
    weighted = ppr_mass_by_step(g, seeds, d, steps)
    ks = list(range(steps))

    panels: list[dict[str, Any]] = [
        {
            "type": "stacked",
            "title": "Mass in each node type per step (weighted graph)",
            "note": "Each bar is one term of x = (1−d) Σ (dPᵀ)ᵏ s. Mass cycles Tag → Question → User every 3 steps while shrinking.",
            "data": {
                "x": ks, "xlabel": "walk step k", "ylabel": "probability mass",
                "series": [
                    {"name": "Tags", "role": "tag", "values": weighted["tag"].tolist()},
                    {"name": "Questions", "role": "question", "values": weighted["question"].tolist()},
                    {"name": "Users", "role": "user", "values": weighted["user"].tolist()},
                ],
            },
        }
    ]
    series = [
        {"name": "Weighted (B4w, κ leak)", "values": weighted_total(weighted).tolist(), "ink": "primary", "dash": False}
    ]
    users_a = ppr_user_scores(g, seeds, d)
    users_b = None
    if ctx.graph_unweighted is not None:
        unweighted = ppr_mass_by_step(ctx.graph_unweighted, seeds, d, steps)
        series.append(
            {"name": "Unweighted (B4, κ=0)", "values": weighted_total(unweighted).tolist(), "ink": "muted", "dash": True}
        )
        users_b = ppr_user_scores(ctx.graph_unweighted, seeds, d)
    panels.append(
        {
            "type": "lines",
            "title": "Total mass still walking, by step",
            "note": "The gap is the absorbing leak: κ and unanswered questions remove mass in the weighted graph.",
            "data": {"x": ks, "xlabel": "walk step k", "ylabel": "mass remaining", "series": series},
        }
    )

    share_a = users_a / max(float(users_a.sum()), 1e-300)
    share_b = users_b / max(float(users_b.sum()), 1e-300) if users_b is not None else None
    top = [int(u) for u in np.argsort(-users_a)[:top_users]]
    rank_b = None
    if users_b is not None:
        order_b = np.argsort(-users_b)
        rank_b = np.empty_like(order_b)
        rank_b[order_b] = np.arange(len(order_b))
    rows = []
    for rnk, u in enumerate(top):
        tip = [["weighted share", f"{share_a[u]:.4f}"]]
        if share_b is not None and rank_b is not None:
            tip += [["unweighted share", f"{share_b[u]:.4f}"], ["unweighted rank", f"#{int(rank_b[u]) + 1}"]]
        rows.append(
            {
                "label": f"#{rnk + 1} user {int(ctx.user_ids[u])}",
                "a": float(share_a[u]),
                "b": float(share_b[u]) if share_b is not None else None,
                "tip": tip,
            }
        )
    panels.append(
        {
            "type": "dumbbell",
            "title": f"Top {len(rows)} users by weighted PPR",
            "note": "Share of total user PPR mass; quality weighting redistributes mass relative to the unweighted graph (H3).",
            "data": {
                "rows": rows, "xlabel": "share of user PPR mass",
                "a_name": "Weighted", "b_name": "Unweighted" if share_b is not None else None,
            },
        }
    )
    return {
        "type": "page",
        "title": f"PPR propagation for query tags {', '.join(tags)} · snapshot {ctx.month}",
        "subtitle": f"Absorbing PPR with d = {d:g}, restart uniform over the query tags.",
        "panels": panels,
        "notes": [],
    }


def weighted_total(by_type: dict[str, np.ndarray]) -> np.ndarray:
    return by_type["tag"] + by_type["question"] + by_type["user"]


# --------------------------------------------------------------------------- structure overview


def _log_bins(values: np.ndarray, n_bins: int = 14) -> list[dict[str, float]]:
    values = values[values > 0]
    edges = np.unique(np.round(np.logspace(0, np.log10(max(values.max(), 2)), n_bins + 1)).astype(int))
    counts, edges_out = np.histogram(values, bins=np.append(edges[:-1], edges[-1] + 1))
    return [
        {"lo": float(lo), "hi": float(hi - 1), "count": int(c)}
        for lo, hi, c in zip(edges_out[:-1], edges_out[1:], counts, strict=True)
    ]


def lap_transition(graph: HeterogeneousGraph) -> sparse.csr_matrix:
    """One full lap `R = P_TQ · P_QU · P_UT` (|T| x |T|): probability a walk from tag s is back at tag t."""
    return (graph.p_tag_to_question @ graph.p_question_to_user @ graph.p_user_to_tag).tocsr()


def overview_payload(ctx: GraphContext, *, heatmap_tags: int = 28) -> dict[str, Any]:
    g = ctx.graph
    p_tq, p_qu = g.p_tag_to_question.tocsr(), g.p_question_to_user.tocsr()
    tag_deg = np.diff(p_tq.indptr)
    user_deg = np.diff(p_qu.tocsc().indptr)
    rowsum = np.asarray(p_qu.sum(axis=1)).ravel()
    answered = np.diff(p_qu.indptr) > 0
    leaks = (1.0 - rowsum[answered]).clip(0.0, 1.0)
    leak_counts, leak_edges = np.histogram(leaks, bins=np.linspace(0.0, 1.0, 21))

    r = lap_transition(g)
    top = np.argsort(-tag_deg)[:heatmap_tags]
    sub = r[top][:, top].toarray()
    from scipy.cluster.hierarchy import leaves_list, linkage

    order = leaves_list(linkage(np.sqrt(sub), method="average"))
    top = top[order]
    sub = sub[np.ix_(order, order)]

    n_edges = int(g.p_tag_to_question.nnz + g.p_question_to_user.nnz + g.p_user_to_tag.nnz)
    tiles = [
        {"label": "Tags", "value": f"{g.num_tags:,}", "role": "tag"},
        {"label": "Questions", "value": f"{g.num_questions:,}", "role": "question"},
        {"label": "Users", "value": f"{g.num_users:,}", "role": "user"},
        {"label": "Edges", "value": f"{n_edges:,}"},
        {"label": "κ leak constant", "value": f"{g.kappa:.3f}"},
        {"label": "Questions with ≥1 answerer", "value": f"{answered.mean():.0%}"},
    ]
    panels = [
        {
            "type": "hist", "role": "tag", "title": "Tag degree",
            "note": "Questions per tag (|Q_t|), log-binned. Tag → Question probability is 1/|Q_t|.",
            "data": {"bins": _log_bins(tag_deg.astype(float)), "xlabel": "questions per tag", "ylabel": "tags", "xlog": True},
        },
        {
            "type": "hist", "role": "user", "title": "User degree",
            "note": "Distinct questions answered per user, log-binned: a heavy tail of prolific answerers.",
            "data": {"bins": _log_bins(user_deg.astype(float)), "xlabel": "questions answered per user", "ylabel": "users", "xlog": True},
        },
        {
            "type": "hist", "role": "question", "title": "Absorbed mass per answered question",
            "note": f"1 − Σ P(q, ·) for the {int(answered.sum()):,} questions with answers. "
            f"A further {int((~answered).sum()):,} questions have no answers and absorb all their mass.",
            "data": {
                "bins": [
                    {"lo": float(lo), "hi": float(hi), "count": int(c)}
                    for lo, hi, c in zip(leak_edges[:-1], leak_edges[1:], leak_counts, strict=True)
                ],
                "xlabel": "absorbed mass at question", "ylabel": "questions", "xlog": False, "decimals": 2,
            },
        },
        {
            "type": "heatmap", "title": f"One-lap tag → tag transition (top {len(top)} tags)",
            "note": "R = P(T→Q) · P(Q→U) · P(U→T): the probability that a walk starting at the row tag is back at the column tag "
            "after one lap. Rows and columns clustered; the diagonal is self-return.",
            "data": {
                "rows": [ctx.tags[int(i)] for i in top], "cols": [ctx.tags[int(i)] for i in top],
                "values": sub.round(6).tolist(),
            },
        },
    ]
    return {
        "type": "page",
        "title": f"Heterogeneous graph structure · snapshot {ctx.month}",
        "subtitle": "Aggregate view of the weighted transition graph used by B4w, B5 and B6.",
        "tiles": tiles, "panels": panels, "notes": [],
    }
