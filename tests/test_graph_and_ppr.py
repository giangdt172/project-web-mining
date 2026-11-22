"""Heterogeneous transition graph and absorbing batched PPR tests (Section 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from so_expert_rec.config import ExperimentConfig
from so_expert_rec.features.quality import compute_answer_quality
from so_expert_rec.features.snapshot import (
    PrecomputedStore,
    build_or_load_graph_snapshot,
    build_or_load_snapshot,
)
from so_expert_rec.graph.builder import (
    build_transition_graph,
    load_heterogeneous_graph,
    save_heterogeneous_graph,
)
from so_expert_rec.graph.ppr import (
    build_per_tag_user_ppr_cache,
    compose_online_ppr,
    ppr_batch,
    ppr_one_hop_diagnostic,
    score_tag_ppr_on_pool,
)
from tests.test_point_in_time_leakage import _build_leakage_fixture


def _build_worked_example_graph(kappa_override: float = 1.0, unweighted: bool = False):
    """Build the exact Section 5.1 worked example (Q1, Q2, plus optional Q3 lone weak answer).

    Tags: 0=pytorch, 1=cuda
    Questions: 0=Q1 (pytorch, cuda), 1=Q2 (pytorch, cuda), 2=Q3 (pytorch; lone weak answer q=0.1)
    Users: 0=Alice, 1=Bob, 2=Charlie (lone weak answer on Q3)
    """
    scores = np.array([15, 7, 8, 0], dtype=np.int32)
    accepted = np.array([1, 0, 0, 0], dtype=np.int8)
    ages = np.array([10.0, 10.0, 5.0, 5.0], dtype=np.float64)
    _, _, q_tau = compute_answer_quality(
        score_m=scores,
        accepted_m=accepted,
        answer_age_days=ages,
        epsilon=0.1,
        beta=1.0,
        tau_days=None,
    )
    # Q1: Alice=3.8726, Bob=2.1794; Q2: Alice=2.2972; Q3: Charlie=0.1
    tq = sparse.csr_matrix(
        np.array(
            [
                [1.0, 1.0, 1.0],  # pytorch -> Q1, Q2, Q3
                [1.0, 1.0, 0.0],  # cuda -> Q1, Q2
            ],
            dtype=np.float64,
        )
    )
    qu = sparse.csr_matrix(
        np.array(
            [
                [q_tau[0], q_tau[1], 0.0],  # Q1 -> Alice, Bob
                [q_tau[2], 0.0, 0.0],       # Q2 -> Alice
                [0.0, 0.0, q_tau[3]],       # Q3 -> Charlie (lone weak)
            ],
            dtype=np.float64,
        )
    )
    ut = sparse.csr_matrix(
        np.array(
            [
                [(q_tau[0] + q_tau[2]) / 2.0, (q_tau[0] + q_tau[2]) / 2.0],  # Alice
                [q_tau[1] / 2.0, q_tau[1] / 2.0],                            # Bob
                [q_tau[3], 0.0],                                              # Charlie
            ],
            dtype=np.float64,
        )
    )
    return build_transition_graph(
        tag_question_adj=tq,
        question_user_qtau=qu,
        user_tag_esum=ut,
        kappa_multiplier=1.0,
        unweighted=unweighted,
        kappa_override=None if unweighted else kappa_override,
    ), q_tau


def test_transition_probabilities_with_kappa_leak(tmp_path: Path) -> None:
    """Verify Q1 -> Alice (0.55), Bob (0.31), leak (0.14); Q2 -> Alice (0.70), leak (0.30); Q3 -> Charlie (0.09)."""
    graph, q_tau = _build_worked_example_graph(kappa_override=1.0, unweighted=False)

    # Check Question -> User transitions & absorbed kappa leaks
    p_qu = graph.p_question_to_user.toarray()
    q1_denom = q_tau[0] + q_tau[1] + 1.0
    assert p_qu[0, 0] == pytest.approx(q_tau[0] / q1_denom, rel=1e-6)
    assert p_qu[0, 1] == pytest.approx(q_tau[1] / q1_denom, rel=1e-6)
    assert round(float(p_qu[0, 0]), 2) == 0.55
    assert round(float(p_qu[0, 1]), 2) == 0.31
    assert round(float(1.0 - p_qu[0].sum()), 2) == 0.14  # kappa leak absorbed

    q2_denom = q_tau[2] + 1.0
    assert p_qu[1, 0] == pytest.approx(q_tau[2] / q2_denom, rel=1e-6)
    assert round(float(p_qu[1, 0]), 2) == 0.70
    assert round(float(1.0 - p_qu[1].sum()), 2) == 0.30

    # Lone weak answer (q = 0.1, kappa = 1.0) gets 0.1 / 1.1 = 0.0909 (~0.09)
    assert p_qu[2, 2] == pytest.approx(0.1 / 1.1, rel=1e-6)
    assert round(float(p_qu[2, 2]), 2) == 0.09

    # Tag -> Question uniform 1 / |Q_t|
    p_tq = graph.p_tag_to_question.toarray()
    assert np.allclose(p_tq[0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(p_tq[1], [0.5, 0.5, 0.0])

    # Every row of P = P_T.T must be sub-stochastic (<= 1.0)
    p_full = graph.p_transpose.T.tocsr()
    row_sums = np.asarray(p_full.sum(axis=1)).ravel()
    assert np.all(row_sums <= 1.0 + 1e-12)

    # Unweighted baseline B4 forces all weights=1 and kappa=0
    graph_u, _ = _build_worked_example_graph(unweighted=True)
    assert graph_u.kappa == 0.0
    p_qu_u = graph_u.p_question_to_user.toarray()
    assert np.allclose(p_qu_u[0], [0.5, 0.5, 0.0])
    assert np.allclose(p_qu_u[1], [1.0, 0.0, 0.0])
    assert np.allclose(p_qu_u[2], [0.0, 0.0, 1.0])

    # Round-trip save & load cache preserves exact graph matrices & metadata
    cache_dir = tmp_path / "graph_cache"
    save_heterogeneous_graph(graph, cache_dir)
    loaded = load_heterogeneous_graph(cache_dir)
    assert loaded.kappa == pytest.approx(graph.kappa)
    assert np.allclose(loaded.p_transpose.toarray(), graph.p_transpose.toarray())


def test_absorbing_ppr_linearity_matches_per_tag_cache() -> None:
    """Verify online composition from cached per-tag vectors equals batched PPR exactly (tags + question seeds)."""
    graph, _ = _build_worked_example_graph(kappa_override=1.0, unweighted=False)
    d = 0.5
    tol = 1e-10

    per_tag_cache = build_per_tag_user_ppr_cache(
        p_transpose=graph.p_transpose,
        tag_slice=graph.tag_slice,
        user_slice=graph.user_slice,
        damping=d,
        tol=tol,
        max_iter=200,
    )
    assert per_tag_cache.shape == (graph.num_tags, graph.num_users)

    # Query 0: tag-only restart (0.7 pytorch, 0.3 cuda)
    # Query 1: mixed tag + question-seed restart (gamma = 0.4 on Q1=0.6, Q2=0.4)
    s_mat = np.zeros((graph.num_nodes, 2), dtype=np.float64)
    s_mat[graph.tag_offset + 0, 0] = 0.7
    s_mat[graph.tag_offset + 1, 0] = 0.3

    s_mat[graph.tag_offset + 0, 1] = 0.6 * 0.25
    s_mat[graph.tag_offset + 1, 1] = 0.6 * 0.75
    s_mat[graph.question_offset + 0, 1] = 0.4 * 0.6
    s_mat[graph.question_offset + 1, 1] = 0.4 * 0.4

    x_batched = ppr_batch(graph.p_transpose, s_mat, damping=d, tol=tol, max_iter=200)
    u_batched_0 = x_batched[graph.user_slice, 0]
    u_batched_1 = x_batched[graph.user_slice, 1]

    u_online_0 = compose_online_ppr(
        tag_weights={0: 0.7, 1: 0.3},
        question_weights=None,
        per_tag_user_ppr=per_tag_cache,
        p_question_to_user=graph.p_question_to_user,
        p_user_to_tag=graph.p_user_to_tag,
        damping=d,
    )
    u_online_1 = compose_online_ppr(
        tag_weights={0: 0.6 * 0.25, 1: 0.6 * 0.75},
        question_weights={0: 0.4 * 0.6, 1: 0.4 * 0.4},
        per_tag_user_ppr=per_tag_cache,
        p_question_to_user=graph.p_question_to_user,
        p_user_to_tag=graph.p_user_to_tag,
        damping=d,
    )

    assert np.allclose(u_online_0, u_batched_0, atol=1e-9, rtol=1e-8)
    assert np.allclose(u_online_1, u_batched_1, atol=1e-9, rtol=1e-8)

    # Alice should strictly outrank Bob and Charlie on pytorch/cuda
    assert u_online_0[0] > u_online_0[1] > u_online_0[2] > 0.0

    # Also verify score_tag_ppr_on_pool matches u_batched_0 on a subset pool
    pi_q0 = sparse.csr_matrix([[0.7, 0.3]], dtype=np.float64)
    pool_scores = score_tag_ppr_on_pool(pi_q0, graph, np.array([2, 0]), damping=d, tol=tol, max_iter=200)
    assert np.allclose(pool_scores[0], [u_batched_0[2], u_batched_0[0]], atol=1e-9)


def test_one_hop_diagnostic_matches_two_step_transition() -> None:
    """Verify PPR_1(u) = sum_t pi_t sum_q P(t, q) P(q, u) and matches 2-step Neumann term."""
    graph, _ = _build_worked_example_graph(kappa_override=1.0, unweighted=False)
    pi = sparse.csr_matrix(
        np.array(
            [
                [0.6, 0.4],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    ppr1 = ppr_one_hop_diagnostic(pi, graph.p_tag_to_question, graph.p_question_to_user).toarray()

    # Explicit manual sum_t pi_t sum_q P(t, q) P(q, u)
    p_tq = graph.p_tag_to_question.toarray()
    p_qu = graph.p_question_to_user.toarray()
    expected = pi.toarray() @ p_tq @ p_qu
    assert np.allclose(ppr1, expected, atol=1e-12)

    # 2-step Neumann term (max_iter=2) of absorbing PPR with tag restart equals d^2 * (1 - d) * PPR_1 on user nodes
    d = 0.5
    s_mat = np.zeros((graph.num_nodes, 2), dtype=np.float64)
    s_mat[graph.tag_slice, :] = pi.toarray().T
    x_2step = ppr_batch(graph.p_transpose, s_mat, damping=d, tol=0.0, max_iter=2)
    u_2step = x_2step[graph.user_slice, :].T
    assert np.allclose(u_2step, (d * d * (1.0 - d)) * ppr1, atol=1e-12)


def test_graph_snapshot_point_in_time_no_leakage(tmp_path: Path) -> None:
    """Verify heterogeneous graph snapshots strictly exclude questions, answers, and votes on/after YYYY-MM-01."""
    _, processed_dir = _build_leakage_fixture(tmp_path)
    store = PrecomputedStore.load(processed_dir)
    cfg = ExperimentConfig()
    snap_dir = tmp_path / "snapshots"

    snap_jul = build_or_load_snapshot("2021-07", cfg, processed_dir, snap_dir, store=store, use_cache=True)
    graph_jul = build_or_load_graph_snapshot(
        "2021-07", cfg, processed_dir, snap_dir, store=store, snapshot=snap_jul, unweighted=False, use_cache=True
    )
    assert graph_jul.num_questions == 1
    u2_idx = store.user_to_idx[2]
    # User 2 only answered on 2021-07-01, so User 2 has zero incoming Q->U and zero outgoing U->T edges in S_{2021-07}
    assert graph_jul.p_question_to_user[:, u2_idx].nnz == 0
    assert graph_jul.p_user_to_tag[u2_idx, :].nnz == 0

    # In S_{2021-08}, Q2 and A20 (user 2) appear in the graph
    snap_aug = build_or_load_snapshot("2021-08", cfg, processed_dir, snap_dir, store=store, use_cache=True)
    graph_aug = build_or_load_graph_snapshot(
        "2021-08", cfg, processed_dir, snap_dir, store=store, snapshot=snap_aug, unweighted=False, use_cache=True
    )
    assert graph_aug.num_questions == 2
    assert graph_aug.p_question_to_user[:, u2_idx].nnz == 1
    assert graph_aug.p_user_to_tag[u2_idx, :].nnz >= 1
