"""Answer quality, recency decay, and expertise matrix tests including Section 5.1 worked example."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from so_expert_rec.features.expertise import (
    build_expertise_matrices,
    build_query_tag_weights,
    score_queries_from_expertise,
)
from so_expert_rec.features.quality import compute_answer_quality


def test_horizon_capped_score_matches_worked_example() -> None:
    """Verify Section 5.1 worked example (no decay, epsilon=0.1, beta=1.0).

    Q1 (pytorch, cuda): Alice (+15, accepted) -> 3.8726, Bob (+7, not accepted) -> 2.1794
    Q2 (pytorch, cuda): Alice (+8, not accepted) -> 2.2972
    E_sum(Alice, pytorch) = (3.8726 + 2.2972) / 2 = 3.0849 (~3.09)
    E_sum(Bob, pytorch)   = 2.1794 / 2 = 1.0897 (~1.09)
    """
    scores = np.array([15, 7, 8], dtype=np.int32)
    accepted = np.array([1, 0, 0], dtype=np.int8)
    ages = np.array([10.0, 10.0, 5.0], dtype=np.float64)

    q, w_tau, q_tau = compute_answer_quality(
        score_m=scores,
        accepted_m=accepted,
        answer_age_days=ages,
        epsilon=0.1,
        beta=1.0,
        tau_days=None,  # no decay
    )
    assert np.allclose(w_tau, 1.0)
    assert q_tau[0] == pytest.approx(0.1 + np.log(16.0) + 1.0, rel=1e-6)
    assert q_tau[1] == pytest.approx(0.1 + np.log(8.0), rel=1e-6)
    assert q_tau[2] == pytest.approx(0.1 + np.log(9.0), rel=1e-6)
    assert round(float(q_tau[0]), 2) == 3.87
    assert round(float(q_tau[1]), 2) == 2.18
    assert round(float(q_tau[2]), 2) == 2.30

    # Users: 0=Alice, 1=Bob; Tags: 0=pytorch, 1=cuda
    answer_user_idx = np.array([0, 1, 0], dtype=np.int32)
    w_at = sparse.csr_matrix(
        np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
    )
    exp = build_expertise_matrices(
        answer_user_idx=answer_user_idx,
        answer_tag_weights=w_at,
        q_tau=q_tau,
        w_tau=w_tau,
        num_users=2,
        smoothing_m=5.0,
        tag_doc_freq=np.array([2.0, 2.0]),
        num_questions=10,
    )
    assert exp.e_sum[0, 0] == pytest.approx((q_tau[0] + q_tau[2]) / 2.0, rel=1e-6)
    assert exp.e_sum[0, 1] == pytest.approx(q_tau[1] / 2.0, rel=1e-6)
    assert round(float(exp.e_sum[0, 0]), 2) == 3.08
    assert round(float(exp.e_sum[0, 1]), 2) == 1.09


def test_answer_level_recency_decay_and_smoothed_e_prod() -> None:
    """Verify decay w_tau is applied once at answer level, E_prod uses n_tau, and IDF handles unseen tags."""
    scores = np.array([10, 10], dtype=np.int32)
    accepted = np.array([0, 0], dtype=np.int8)
    ages = np.array([0.0, 365.0], dtype=np.float64)

    q, w_tau, q_tau = compute_answer_quality(
        score_m=scores,
        accepted_m=accepted,
        answer_age_days=ages,
        epsilon=0.1,
        beta=1.0,
        tau_days=365.0,
    )
    assert w_tau[0] == pytest.approx(1.0)
    assert w_tau[1] == pytest.approx(np.exp(-1.0))
    assert q_tau[1] == pytest.approx(q[1] * np.exp(-1.0))

    # User 0 has recent answer on tag 0; User 1 has old answer on tag 0; tag 1 unseen
    w_at = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64))
    exp = build_expertise_matrices(
        answer_user_idx=np.array([0, 1], dtype=np.int32),
        answer_tag_weights=w_at,
        q_tau=q_tau,
        w_tau=w_tau,
        num_users=2,
        smoothing_m=5.0,
        tag_doc_freq=np.array([4.0, 0.0]),
        num_questions=20,
    )
    # Tag 0 prior mean mu_0 equals q[0] since both answers have identical q and differ only in w_tau
    assert exp.mu_t[0] == pytest.approx(q[0], rel=1e-6)
    assert exp.mu_t[1] == 0.0
    assert exp.tag_idf[0] == pytest.approx(np.log(20.0 / 4.0))
    assert exp.tag_idf[1] == 0.0

    # E_prod should rank User 0 (recent, larger n_tau) strictly above User 1 (old, smaller n_tau)
    assert exp.e_prod[0, 0] > exp.e_prod[0, 1] > 0.0
    assert exp.e_prod[1, 0] == 0.0

    # Query has seen tag 0 and unseen tag 1 -> Pi puts 100% weight on tag 0, 0 on unseen tag 1
    q_tags = sparse.csr_matrix(np.array([[1.0, 1.0]], dtype=np.float64))
    pi = build_query_tag_weights(q_tags, exp.tag_idf, uniform=False)
    assert pi[0, 0] == pytest.approx(1.0)
    assert pi[0, 1] == 0.0
    scored = score_queries_from_expertise(pi, exp.e_prod).toarray()
    assert scored[0, 0] == pytest.approx(exp.e_prod[0, 0])
    assert scored[0, 1] == pytest.approx(exp.e_prod[0, 1])
