"""Graph visualization payloads and renderers (viz package)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so_expert_rec.graph.ppr import ppr_batch
from so_expert_rec.viz import render_html, render_layered, render_overview, render_ppr
from so_expert_rec.viz.data import (
    GraphContext,
    ego_payload,
    overview_payload,
    ppr_mass_by_step,
    ppr_payload,
    worked_example_payload,
)


def _edge(payload: dict, s: str, t: str) -> dict:
    (e,) = [e for e in payload["edges"] if e["s"] == s and e["t"] == t]
    return e


def _example_context() -> GraphContext:
    _, graph = worked_example_payload()
    return GraphContext(
        month="2021-12",
        graph=graph,
        graph_unweighted=graph,
        tags=["pytorch", "cuda"],
        user_ids=np.array([101, 102, 103]),
        question_ids=np.array([1, 2, 3]),
        question_titles={1: "CUDA OOM", 2: "GPU memory leak", 3: "lone weak answer"},
        damping=0.5,
    )


def test_worked_example_matches_section_5_1() -> None:
    payload, graph = worked_example_payload()
    assert round(_edge(payload, "q0", "u0")["w"], 2) == 0.55  # Q1 -> Alice
    assert round(_edge(payload, "q0", "u1")["w"], 2) == 0.31  # Q1 -> Bob
    assert round(_edge(payload, "q1", "u0")["w"], 2) == 0.70  # Q2 -> Alice
    assert round(_edge(payload, "q2", "u2")["w"], 2) == 0.09  # lone weak answer
    assert round(_edge(payload, "q0", "sink")["w"], 2) == 0.14  # kappa leak
    assert round(_edge(payload, "q1", "sink")["w"], 2) == 0.30
    assert graph.kappa == pytest.approx(1.0)
    assert any(n["kind"] == "sink" for n in payload["nodes"])


def test_layered_payload_is_consistent() -> None:
    payload, _ = worked_example_payload()
    ids = {n["id"] for n in payload["nodes"]}
    for e in payload["edges"]:
        assert e["s"] in ids and e["t"] in ids
        assert 0.0 <= e["wn"] <= 1.0 and e["w"] > 0.0
    for col in range(len(payload["columns"])):
        orders = sorted(n["order"] for n in payload["nodes"] if n["col"] == col)
        assert orders == [float(i) for i in range(len(orders))]  # dense, unique per column


def test_ego_payload_edges_come_straight_from_the_graph() -> None:
    ctx = _example_context()
    payload = ego_payload(ctx, "pytorch", top_users=3, top_questions=3)
    p_qu = ctx.graph.p_question_to_user.toarray()
    p_tq = ctx.graph.p_tag_to_question.toarray()
    for e in payload["edges"]:
        if e["kind"] != "walk":
            continue
        if e["layer"] == "question->user":
            assert e["w"] == pytest.approx(p_qu[int(e["s"][1:]), int(e["t"][1:])])
        if e["layer"] == "tag->question":
            assert e["w"] == pytest.approx(p_tq[int(e["s"][1:]), int(e["t"][1:])])
    with pytest.raises(KeyError):
        ego_payload(ctx, "not-a-tag")


def test_ppr_steps_sum_to_exact_ppr() -> None:
    ctx = _example_context()
    g = ctx.graph
    steps = ppr_mass_by_step(g, {0: 1.0}, damping=0.5, steps=80)
    total = sum(v.sum() for v in steps.values())
    restart = np.zeros((g.num_nodes, 1))
    restart[g.tag_offset + 0, 0] = 1.0
    exact = ppr_batch(g.p_transpose, restart, damping=0.5, tol=1e-15, max_iter=500)[:, 0]
    assert total == pytest.approx(exact.sum(), rel=1e-9)
    assert steps["tag"][0] == pytest.approx(0.5)  # (1 - d) restart mass starts in the tag block
    page = ppr_payload(ctx, ["pytorch"], steps=6, top_users=2)
    assert [p["type"] for p in page["panels"]] == ["stacked", "lines", "dumbbell"]


def test_overview_payload_shapes() -> None:
    ctx = _example_context()
    page = overview_payload(ctx, heatmap_tags=2)
    kinds = [p["type"] for p in page["panels"]]
    assert kinds == ["hist", "hist", "hist", "heatmap"]
    heat = page["panels"][3]["data"]
    assert np.asarray(heat["values"]).shape == (2, 2)


def test_renderers_write_files_and_escape_script_close(tmp_path: Path) -> None:
    payload, _ = worked_example_payload()
    payload["nodes"][0]["label"] = "</script><b>x</b>"
    html_path = render_html(payload, tmp_path / "w.html")
    text = html_path.read_text(encoding="utf-8")
    assert text.count("</script>") == 2  # payload block + main script only; label cannot break out
    assert "<\\/script>" in text
    assert render_layered(payload, tmp_path / "w.png").stat().st_size > 1000

    ctx = _example_context()
    assert render_ppr(ppr_payload(ctx, ["pytorch"], steps=6, top_users=3), tmp_path / "p.png").exists()
    assert render_overview(overview_payload(ctx, heatmap_tags=2), tmp_path / "o.png").exists()
    assert render_html(overview_payload(ctx, heatmap_tags=2), tmp_path / "o.html").exists()


def test_explorer_payload_roundtrips_the_graph(tmp_path: Path) -> None:
    import base64

    from so_expert_rec.viz.interactive import explorer_payload, render_explorer

    ctx = _example_context()
    payload = explorer_payload(ctx, default_tags=["pytorch"], popular=["pytorch", "cuda"])
    qu = payload["graphs"]["weighted"]["qu"]
    ip = np.frombuffer(base64.b64decode(qu["ip"]), dtype="<i4")
    ix = np.frombuffer(base64.b64decode(qu["ix"]), dtype="<i4")
    dt = np.frombuffer(base64.b64decode(qu["dt"]), dtype="<f4")
    dense = np.zeros(qu["shape"])
    for r in range(qu["shape"][0]):
        dense[r, ix[ip[r] : ip[r + 1]]] = dt[ip[r] : ip[r + 1]]
    np.testing.assert_allclose(dense, ctx.graph.p_question_to_user.toarray(), rtol=1e-6)
    assert set(payload["graphs"]) == {"weighted", "unweighted"}

    payload["tags"][0] = "</script><img src=x onerror=alert(1)>"
    text = render_explorer(payload, tmp_path / "e.html").read_text(encoding="utf-8")
    assert text.count("</script>") == 2  # payload block + main script; tag names cannot break out
    assert "/*__KIT" not in text and "__PAYLOAD__" not in text
