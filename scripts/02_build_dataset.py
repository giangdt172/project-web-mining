#!/usr/bin/env python3
"""Milestone 2 prep: Precompute cumulative votes, answer tag weights, B1 reputation events, and initial eval queries."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from so_expert_rec.config import load_config
from so_expert_rec.data.extract import resolve_synonyms_source
from so_expert_rec.data.precompute import (
    build_initial_queries_table,
    precompute_answer_tag_weights,
    precompute_cumulative_votes,
    precompute_reputation_events,
)
from so_expert_rec.data.tag_cluster import build_synonym_map


def run_build_dataset(config_path: Path) -> None:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    interim_dir = cfg.paths.interim_dir
    processed_dir = cfg.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    questions_pq = interim_dir / "questions.parquet"
    answers_pq = interim_dir / "answers.parquet"
    votes_pq = interim_dir / "votes.parquet"
    ph_pq = interim_dir / "post_history.parquet"

    for req in (questions_pq, answers_pq, votes_pq):
        if not req.exists():
            raise FileNotFoundError(f"Missing Milestone 1 output: {req}. Run scripts/01_extract_and_validate.py first.")

    syn_src = interim_dir / "tag_synonyms.parquet"
    if not syn_src.exists():
        syn_src_raw = resolve_synonyms_source(cfg.paths.raw_dir)
        syn_map = build_synonym_map(syn_src_raw)
    else:
        syn_map = build_synonym_map(syn_src)

    print("[1/4] Precomputing sorted eligible answers and cumulative vote arrays...")
    t_step = time.perf_counter()
    precompute_cumulative_votes(
        answers_parquet=answers_pq,
        votes_parquet=votes_pq,
        output_path=processed_dir / "cumulative_votes.npz",
        answers_index_path=processed_dir / "answers_index.parquet",
    )
    print(f"      Done in {time.perf_counter() - t_step:.2f}s")

    print("[2/4] Precomputing per-answer tag weights w_{a,t} and vocab.json...")
    t_step = time.perf_counter()
    precompute_answer_tag_weights(
        questions_parquet=questions_pq,
        answers_index_parquet=processed_dir / "answers_index.parquet",
        output_path=processed_dir / "answer_tag_weights.npz",
        vocab_path=processed_dir / "vocab.json",
    )
    print(f"      Done in {time.perf_counter() - t_step:.2f}s")

    print("[3/4] Precomputing point-in-time vote-based reputation events (B1)...")
    t_step = time.perf_counter()
    precompute_reputation_events(
        questions_parquet=questions_pq,
        answers_parquet=answers_pq,
        votes_parquet=votes_pq,
        vocab_path=processed_dir / "vocab.json",
        output_path=processed_dir / "reputation_votes.npz",
    )
    print(f"      Done in {time.perf_counter() - t_step:.2f}s")

    print("[4/4] Building initial PostHistory query table (BM25 + dense views)...")
    t_step = time.perf_counter()
    build_initial_queries_table(
        questions_parquet=questions_pq,
        post_history_parquet=ph_pq if ph_pq.exists() else None,
        synonym_map=syn_map,
        output_path=processed_dir / "eval_queries.parquet",
    )
    print(f"      Done in {time.perf_counter() - t_step:.2f}s")
    print(f"Dataset precomputation complete in {time.perf_counter() - t0:.2f}s -> {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    run_build_dataset(args.config)


if __name__ == "__main__":
    main()
