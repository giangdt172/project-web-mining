"""Normalization, fusion, deterministic tie-breaking, metrics, and significance tests (Sections 7, 9)."""

from __future__ import annotations

import numpy as np
import pytest

from so_expert_rec.evaluation.metrics import (
    compute_load_concentration,
    evaluate_ranked_user_ids,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from so_expert_rec.evaluation.significance import (
    apply_holm_correction,
    paired_bootstrap_ci,
    paired_randomization_test,
)
from so_expert_rec.models.fusion import (
    fuse_linear,
    fuse_rrf,
    make_tie_break_keys,
    normalize_component_scores,
    rank_candidates_with_tie_break,
)


def test_normalization_handles_tiny_ppr_values_and_zeros() -> None:
    """Verify median-scaled log1p + z-score normalization handles tiny PPR values & all-zero rows."""
    scores = np.array(
        [
            [0.0, 1e-9, 2e-9, 1e-8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    norm = normalize_component_scores(scores)
    assert np.all(np.isfinite(norm))
    # Row 0: zero mean, unit std, strictly preserves order
    assert norm[0].mean() == pytest.approx(0.0, abs=1e-12)
    assert norm[0].std() == pytest.approx(1.0, abs=1e-12)
    assert norm[0, 3] > norm[0, 2] > norm[0, 1] > norm[0, 0]
    # Row 1: all-zero stays all-zero without NaNs
    assert np.allclose(norm[1], 0.0)

    # Verify linear fusion & RRF combine components cleanly
    c1 = np.array([[1.0, 3.0, 2.0]])
    c2 = np.array([[10.0, 20.0, 30.0]])
    lin = fuse_linear({"a": normalize_component_scores(c1), "b": normalize_component_scores(c2)}, {"a": 0.5, "b": 0.5})
    assert lin.shape == (1, 3)
    rrf = fuse_rrf({"a": c1, "b": c2}, rrf_k=60)
    # Candidate 1 is rank 1 in c1 and rank 2 in c2 -> 1/61 + 1/62
    assert rrf[0, 1] == pytest.approx(1.0 / 61.0 + 1.0 / 62.0)


def test_tie_breaking_is_deterministic_and_shared_across_systems() -> None:
    """Verify tied users are ordered identically across systems and askers are always excluded."""
    qids = np.array([101, 202], dtype=np.int64)
    keys_sys_a = make_tie_break_keys(qids, num_pool_users=5, seed=42)
    keys_sys_b = make_tie_break_keys(qids, num_pool_users=5, seed=42)
    assert np.array_equal(keys_sys_a, keys_sys_b)
    assert not np.array_equal(keys_sys_a[0], keys_sys_a[1])

    # All users tied at 0.0; asker is at pool position 2 for Q101, -1 (not in pool) for Q202
    tied_scores = np.zeros((2, 5), dtype=np.float64)
    asker_pos = np.array([2, -1], dtype=np.int64)
    ranked_a = rank_candidates_with_tie_break(tied_scores, asker_pos, keys_sys_a, top_k=4)
    ranked_b = rank_candidates_with_tie_break(tied_scores, asker_pos, keys_sys_b, top_k=4)

    assert np.array_equal(ranked_a, ranked_b)
    # Asker (pos 2) must NEVER appear in top-4 for Q101
    assert 2 not in ranked_a[0]


def test_graded_ndcg_gini_and_holm_correction() -> None:
    """Verify graded NDCG@10, Recall/MRR, out-of-pool ceiling, Gini/top-1%, bootstrap, randomization, Holm."""
    # Matrix form
    ranked = np.array([[0, 1, 2, 3]], dtype=np.int64)
    graded = np.array([[1, 2, 0, 0]], dtype=np.int64)
    expected_dcg = (2**1 - 1) / np.log2(2) + (2**2 - 1) / np.log2(3)
    expected_idcg = (2**2 - 1) / np.log2(2) + (2**1 - 1) / np.log2(3)
    assert ndcg_at_k(ranked, graded, k=10)[0] == pytest.approx(expected_dcg / expected_idcg)
    assert recall_at_k(ranked, graded > 0, k=1)[0] == pytest.approx(0.5)
    assert recall_at_k(ranked, graded > 0, k=2)[0] == pytest.approx(1.0)
    assert mrr_at_k(ranked, graded > 0, k=10)[0] == pytest.approx(1.0)

    # Full ground-truth ceiling form: user 999 is accepted (rel=2) but NOT in ranked list (cold-start)
    metrics = evaluate_ranked_user_ids(
        ranked_user_ids=np.array([[10, 20, 30]], dtype=np.int64),
        query_ground_truth=[{10: 1, 999: 2}],
        k_ndcg=10,
        k_recall=(5, 10),
        k_mrr=10,
    )
    # Recall must be 0.5 (1 of 2 positives found), NOT 1.0!
    assert metrics["recall@5"][0] == pytest.approx(0.5)
    # NDCG uses full IDCG (rel 2 at rank 1, rel 1 at rank 2)
    assert metrics["ndcg@10"][0] == pytest.approx(1.0 / expected_idcg)

    # Load concentration: all 100 slots go to user 1 out of 100 pool users
    concentrated = np.ones((20, 5), dtype=np.int64)
    top1_share, gini = compute_load_concentration(concentrated, num_pool_users=100)
    assert top1_share == pytest.approx(1.0)
    assert gini == pytest.approx(0.99, abs=1e-6)

    # Uniform over 100 users -> Gini == 0, top 1% share == 0.01
    uniform = np.arange(100, dtype=np.int64).reshape(20, 5)
    top1_u, gini_u = compute_load_concentration(uniform, num_pool_users=100)
    assert top1_u == pytest.approx(0.01)
    assert gini_u == pytest.approx(0.0, abs=1e-12)

    # Significance: clearly better system A vs B
    rng = np.random.default_rng(7)
    b = rng.uniform(0.1, 0.3, size=80)
    a = b + 0.15
    delta, ci_lo, ci_hi = paired_bootstrap_ci(a, b, n_samples=500, alpha=0.05, seed=7)
    assert delta == pytest.approx(0.15)
    assert ci_lo > 0.0 and ci_hi >= ci_lo
    p_val = paired_randomization_test(a, b, n_trials=1000, seed=7)
    assert p_val < 0.01

    # Holm-Bonferroni correction across 4 hypotheses
    holm = apply_holm_correction({"H1": 0.005, "H2": 0.02, "H3": 0.04, "H4": 0.50}, alpha=0.05)
    assert holm["H1"] == (pytest.approx(0.02), True)    # 0.005 * 4 = 0.02 < 0.05
    assert holm["H2"] == (pytest.approx(0.06), False)   # 0.02 * 3 = 0.06 >= 0.05 -> stops rejection
    assert holm["H3"] == (pytest.approx(0.08), False)   # max(0.06, 0.04 * 2) = 0.08
    assert holm["H4"] == (pytest.approx(0.50), False)
