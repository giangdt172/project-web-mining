"""BM25 per-snapshot retrieval, question-seeded PPR, TextSim, and B6 tests (Section 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from so_expert_rec.config import ExperimentConfig
from so_expert_rec.features.snapshot import (
    PrecomputedStore,
    build_or_load_bm25_snapshot,
    build_or_load_snapshot,
)
from so_expert_rec.graph.ppr import build_per_tag_user_ppr_cache, compose_online_ppr
from so_expert_rec.models.baselines import (
    score_b6_fusion,
    score_b6_ppr_with_text,
    score_b6_text_sim,
)
from so_expert_rec.retrieval.bm25 import build_bm25_index, load_bm25_index, retrieve_bm25_batch
from so_expert_rec.retrieval.hybrid import (
    build_restart_matrix_with_text_seeds,
    compute_text_sim_scores,
    fuse_retrieval_rrf,
)
from tests.test_graph_and_ppr import _build_worked_example_graph
from tests.test_point_in_time_leakage import _build_leakage_fixture


def test_bm25_index_retrieve_save_load_and_empty_queries(tmp_path: Path) -> None:
    """Verify bm25s index ranking, zero-score masking (-1), save/load round-trip, and empty edge cases."""
    corpus = [
        "pytorch cuda out of memory error during backward pass RuntimeError",
        "keras sequential model fit validation loss nan",
        "tensorflow dataset prefetch pipeline performance",
    ]
    idx_dir = tmp_path / "bm25_cache"
    idx = build_bm25_index(corpus, index_dir=idx_dir)
    assert idx.num_docs == 3

    queries = ["cuda out of memory RuntimeError", "unmatchedxyztoken", ""]
    docs, scores = retrieve_bm25_batch(idx, queries, top_k=5)
    assert docs.shape == (3, 3)  # clamped to num_docs=3
    assert docs[0, 0] == 0
    assert scores[0, 0] > 0.0
    # Unmatched and empty queries must be masked to -1 and 0.0
    assert np.all(docs[1] == -1) and np.all(scores[1] == 0.0)
    assert np.all(docs[2] == -1) and np.all(scores[2] == 0.0)

    reloaded = load_bm25_index(idx_dir)
    docs2, scores2 = retrieve_bm25_batch(reloaded, queries, top_k=5)
    assert np.array_equal(docs, docs2)
    assert np.allclose(scores, scores2)

    # Empty corpus edge case
    empty_idx = build_bm25_index([], index_dir=tmp_path / "empty_bm25")
    ed, es = retrieve_bm25_batch(empty_idx, ["cuda"], top_k=5)
    assert ed.shape == (1, 0) and es.shape == (1, 0)
    assert load_bm25_index(tmp_path / "empty_bm25").num_docs == 0


def test_hybrid_rrf_restart_seeds_and_text_sim() -> None:
    """Verify RRF retrieval fusion, mixed tag+question restart normalization, TextSim, and online PPR match."""
    # 1. RRF fusion
    bm25_idx = np.array([[2, 0, 1]], dtype=np.int64)
    dense_idx = np.array([[0, 2, 3]], dtype=np.int64)
    fused_idx, fused_sc = fuse_retrieval_rrf(bm25_idx, dense_idx, top_k=3, rrf_k=60)
    # Docs 0 and 2 both get 1/61 + 1/62; deterministic tie-break orders doc 0 before doc 2
    assert list(fused_idx[0]) == [0, 2, 1]
    assert fused_sc[0, 0] == pytest.approx(1.0 / 61.0 + 1.0 / 62.0)
    assert fused_sc[0, 1] == pytest.approx(1.0 / 61.0 + 1.0 / 62.0)
    assert fused_sc[0, 2] == pytest.approx(1.0 / 63.0)

    # 2. Mixed tag + question seeds on Section 5.1 worked example graph
    graph, q_tau = _build_worked_example_graph(kappa_override=1.0, unweighted=False)
    pi = sparse.csr_matrix(
        np.array(
            [
                [0.6, 0.4],  # Q0: normal tags + retrieved questions
                [1.0, 0.0],  # Q1: normal tag, NO positive retrieved questions -> fallback to 100% tag
                [0.0, 0.0],  # Q2: unseen tags, positive retrieved question -> fallback to 100% question
            ],
            dtype=np.float64,
        )
    )
    ret_idx = np.array([[0, 1], [-1, -1], [0, -1]], dtype=np.int64)
    ret_sc = np.array([[3.0, 1.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    restart = build_restart_matrix_with_text_seeds(
        pi_matrix=pi,
        retrieved_q_indices=ret_idx,
        retrieved_q_scores=ret_sc,
        tag_offset=graph.tag_offset,
        question_offset=graph.question_offset,
        num_nodes=graph.num_nodes,
        gamma=0.4,
    ).toarray()
    # Every restart column must sum to 1.0
    assert np.allclose(restart.sum(axis=0), [1.0, 1.0, 1.0])
    # Col 0: tags scaled by 0.6, Q0/Q1 scaled by 0.4 * (3/4, 1/4)
    assert restart[graph.tag_offset + 0, 0] == pytest.approx(0.6 * 0.6)
    assert restart[graph.question_offset + 0, 0] == pytest.approx(0.4 * 0.75)
    assert restart[graph.question_offset + 1, 0] == pytest.approx(0.4 * 0.25)
    # Col 1: no positive retrieved docs -> 100% tag 0
    assert restart[graph.tag_offset + 0, 1] == pytest.approx(1.0)
    # Col 2: no seen tags -> 100% Q0
    assert restart[graph.question_offset + 0, 2] == pytest.approx(1.0)

    # 3. TextSim exact manual check
    qu_qtau = sparse.csr_matrix(
        np.array(
            [
                [q_tau[0], q_tau[1], 0.0],
                [q_tau[2], 0.0, 0.0],
                [0.0, 0.0, q_tau[3]],
            ],
            dtype=np.float64,
        )
    )
    ts = compute_text_sim_scores(ret_idx, ret_sc, qu_qtau).toarray()
    assert ts[0, 0] == pytest.approx(3.0 * q_tau[0] + 1.0 * q_tau[2])  # Alice
    assert ts[0, 1] == pytest.approx(3.0 * q_tau[1])                   # Bob
    assert ts[0, 2] == 0.0                                             # Charlie
    assert np.allclose(ts[1], 0.0)

    # 4. Online compose_online_ppr matches score_b6_ppr_with_text to machine precision
    pool = np.array([0, 1, 2], dtype=np.int64)
    batched = score_b6_ppr_with_text(pi, ret_idx, ret_sc, graph, pool, gamma=0.4, damping=0.5, tol=1e-10)
    cache = build_per_tag_user_ppr_cache(graph.p_transpose, graph.tag_slice, graph.user_slice, damping=0.5, tol=1e-10)
    online_q0 = compose_online_ppr(
        tag_weights={0: 0.6 * 0.6, 1: 0.6 * 0.4},
        question_weights={0: 0.4 * 0.75, 1: 0.4 * 0.25},
        per_tag_user_ppr=cache,
        p_question_to_user=graph.p_question_to_user,
        p_user_to_tag=graph.p_user_to_tag,
        damping=0.5,
    )
    assert np.allclose(online_q0, batched[0], atol=1e-9, rtol=1e-8)

    # 5. B6 fusion smoke check (both linear and RRF)
    ts_pool = score_b6_text_sim(ret_idx, ret_sc, qu_qtau, pool)
    b3_dummy = np.array([[3.0, 1.0, 0.1], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    fused_lin = score_b6_fusion(b3_dummy, batched, ts_pool, method="linear")
    fused_rrf = score_b6_fusion(b3_dummy, batched, ts_pool, method="rrf", rrf_k=60)
    assert fused_lin.shape == (3, 3) and np.all(np.isfinite(fused_lin))
    assert fused_rrf.shape == (3, 3) and np.all(np.isfinite(fused_rrf))
    # Alice (user 0) ranks #1 on Query 0 in both modes
    assert int(np.argmax(fused_lin[0])) == 0
    assert int(np.argmax(fused_rrf[0])) == 0


def test_bm25_snapshot_point_in_time_no_leakage(tmp_path: Path) -> None:
    """Verify BM25 snapshot S_{2021-07} strictly indexes < 2021-07-01 questions (Q2 excluded until 2021-08)."""
    _, processed_dir = _build_leakage_fixture(tmp_path)
    store = PrecomputedStore.load(processed_dir)
    cfg = ExperimentConfig()
    snap_dir = tmp_path / "snapshots"

    snap_jul = build_or_load_snapshot("2021-07", cfg, processed_dir, snap_dir, store=store, use_cache=True)
    bm25_jul = build_or_load_bm25_snapshot("2021-07", processed_dir, snap_dir, store=store, snapshot=snap_jul)
    assert bm25_jul.num_docs == 1  # Strictly Q1 only; Q2 on 2021-07-01 must NOT be indexed!

    # Searching for Q2's unique initial text ("Initial title") must return NO match in S_{2021-07}
    docs_jul, scores_jul = retrieve_bm25_batch(bm25_jul, ["Initial title"], top_k=5)
    assert np.all(docs_jul == -1) and np.all(scores_jul == 0.0)

    # One month later (S_{2021-08}): Q2 IS in the corpus and matches "Initial title"
    snap_aug = build_or_load_snapshot("2021-08", cfg, processed_dir, snap_dir, store=store, use_cache=True)
    bm25_aug = build_or_load_bm25_snapshot("2021-08", processed_dir, snap_dir, store=store, snapshot=snap_aug)
    assert bm25_aug.num_docs == 2
    docs_aug, scores_aug = retrieve_bm25_batch(bm25_aug, ["Initial title"], top_k=5)
    assert docs_aug[0, 0] == 1 and scores_aug[0, 0] > 0.0
