#!/usr/bin/env python3
"""Milestone 1: Stream-parse XML/.7z dump to Parquet, score sanity check, horizon curves, volume check."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from so_expert_rec.config import load_config
from so_expert_rec.data.extract import (
    collect_eval_question_ids,
    collect_kept_post_ids,
    extract_post_history_xml,
    extract_posts_xml,
    extract_tags_and_synonyms,
    extract_votes_xml,
    resolve_synonyms_source,
    resolve_table_source,
    run_vote_score_sanity_check,
    scan_question_tags,
)
from so_expert_rec.data.labels import (
    build_fixed_horizon_labels,
    check_monthly_volumes,
    compute_horizon_curves,
)
from so_expert_rec.data.tag_cluster import build_synonym_map


def run_milestone_1(
    config_path: Path,
    expand_cluster: bool = True,
    skip_post_history: bool = False,
) -> dict[str, object]:
    """Execute Milestone 1 end-to-end and write diagnostics to `reports/`."""
    t_start = time.perf_counter()
    cfg = load_config(config_path)

    raw_dir = cfg.paths.raw_dir
    interim_dir = cfg.paths.interim_dir
    processed_dir = cfg.paths.processed_dir
    figures_dir = cfg.paths.reports_dir / "figures"
    tables_dir = cfg.paths.reports_dir / "tables"

    for d in (interim_dir, processed_dir, figures_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    posts_src = resolve_table_source(raw_dir, "Posts")
    votes_src = resolve_table_source(raw_dir, "Votes")
    ph_src = resolve_table_source(raw_dir, "PostHistory")
    tags_src = resolve_table_source(raw_dir, "Tags")
    syn_src = resolve_synonyms_source(raw_dir)

    if posts_src is None:
        raise FileNotFoundError(
            f"Could not find Posts.xml or a .7z archive containing Posts.xml in {raw_dir}."
        )
    if votes_src is None:
        raise FileNotFoundError(
            f"Could not find Votes.xml or a .7z archive containing Votes.xml in {raw_dir}."
        )

    print(f"[1/6] Loading optional tag synonyms from: {syn_src or 'none'}")
    synonym_map = build_synonym_map(syn_src)
    print(f"      Loaded {len(synonym_map)} canonical tag synonym mappings.")

    print(f"[2/6] Pass 1: scanning question tags in {posts_src[0]} (expand={expand_cluster})...")
    scan_res = scan_question_tags(
        posts_source=posts_src[0],
        seed_tags=cfg.tag_cluster.seed_tags,
        synonym_map=synonym_map,
        cooccurrence_threshold=cfg.tag_cluster.cooccurrence_threshold,
        min_tag_count=cfg.tag_cluster.min_tag_count,
        inner_filename=posts_src[1],
        expand_cluster=expand_cluster,
    )
    cluster_tags = set(scan_res["cluster_tags"])  # type: ignore[arg-type]
    kept_qids = set(scan_res["kept_question_ids"])  # type: ignore[arg-type]
    print(
        f"      Scanned {scan_res['total_questions_scanned']:,} questions -> "
        f"{len(cluster_tags)} cluster tags -> {len(kept_qids):,} kept questions."
    )

    cluster_meta_path = processed_dir / "cluster_tags.json"
    cluster_meta_path.write_text(
        json.dumps(
            {
                "seed_tags": scan_res["seed_tags"],
                "cluster_tags": sorted(cluster_tags),
                "cooccurrence_threshold": cfg.tag_cluster.cooccurrence_threshold,
                "min_tag_count": cfg.tag_cluster.min_tag_count,
                "total_questions_scanned": scan_res["total_questions_scanned"],
                "kept_questions": len(kept_qids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    audit_rows = scan_res["audit_rows"]
    if audit_rows:
        with (tables_dir / "tag_cluster_audit.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "tag",
                    "is_seed",
                    "total_questions",
                    "seed_cooccur_questions",
                    "p_seed_given_tag",
                    "selected",
                ],
            )
            writer.writeheader()
            writer.writerows(audit_rows)  # type: ignore[arg-type]

    print(f"[3/6] Pass 2: extracting filtered Posts, Votes, Tags, and PostHistory to {interim_dir}...")
    questions_parquet, answers_parquet = extract_posts_xml(
        xml_path=posts_src[0],
        output_dir=interim_dir,
        cluster_tags=cluster_tags,
        synonym_map=synonym_map,
        kept_question_ids=kept_qids,
        inner_filename=posts_src[1],
    )
    kept_post_ids = collect_kept_post_ids(questions_parquet, answers_parquet)
    votes_parquet = extract_votes_xml(
        xml_path=votes_src[0],
        kept_post_ids=kept_post_ids,
        output_path=interim_dir / "votes.parquet",
        inner_filename=votes_src[1],
    )
    extract_tags_and_synonyms(
        tags_xml=tags_src[0] if tags_src else None,
        synonyms_path=syn_src,
        output_dir=interim_dir,
        tags_inner_filename=tags_src[1] if tags_src else None,
    )

    post_history_parquet: Path | None = None
    if not skip_post_history and ph_src is not None:
        eval_qids = collect_eval_question_ids(
            questions_parquet=questions_parquet,
            eval_start=cfg.splits.val_tune_start,
        )
        print(f"      Extracting PostHistory (types 1/2/3/10) for {len(eval_qids):,} eval questions...")
        post_history_parquet = extract_post_history_xml(
            xml_path=ph_src[0],
            eval_question_ids=eval_qids,
            output_path=interim_dir / "post_history.parquet",
            include_close_events=True,
            inner_filename=ph_src[1],
        )
    else:
        print("      Skipping PostHistory extraction (file absent or --skip-post-history set).")

    print("[4/6] Running vote-reconstructed score sanity check...")
    sanity = run_vote_score_sanity_check(questions_parquet, answers_parquet, votes_parquet)
    (tables_dir / "score_sanity_check.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(
        f"      Posts checked: {int(sanity['total_posts']):,} | "
        f"exact match: {sanity['exact_match_rate']:.2%} | "
        f"within +-1: {sanity['within_1_rate']:.2%} | "
        f"MAE: {sanity['mae']:.4f}"
    )

    print("[5/6] Computing answer-delay and vote-delay horizon curves...")
    horizons = compute_horizon_curves(
        questions_parquet=questions_parquet,
        answers_parquet=answers_parquet,
        votes_parquet=votes_parquet,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    ans_cap = horizons["answer_capture_by_day"]
    vote_cap = horizons["vote_capture_by_day"]
    print(
        f"      Answer capture @ 7d: {ans_cap.get(7, 0.0):.2%} | "  # type: ignore[union-attr]
        f"Vote capture @ 30d: {vote_cap.get(30, 0.0):.2%}"  # type: ignore[union-attr]
    )

    print("[6/6] Building fixed-horizon labels and checking monthly query volumes...")
    labels_parquet = build_fixed_horizon_labels(
        questions_parquet=questions_parquet,
        answers_parquet=answers_parquet,
        votes_parquet=votes_parquet,
        post_history_parquet=post_history_parquet,
        answer_horizon_days=cfg.labels.answer_horizon_days,
        vote_horizon_days=cfg.labels.vote_horizon_days,
        useful_score_threshold=cfg.labels.useful_score_threshold,
        dump_date=cfg.labels.dump_date,
        output_path=processed_dir / "labels.parquet",
    )
    volumes = check_monthly_volumes(
        questions_parquet=questions_parquet,
        labels_parquet=labels_parquet,
        splits=cfg.splits,
        post_history_parquet=post_history_parquet,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )

    splits_summary = volumes["splits"]
    if isinstance(splits_summary, dict) and splits_summary:
        print("      Split evaluable query counts (non-duplicate, >=1 useful horizon answer):")
        for split_name, stats in splits_summary.items():
            print(
                f"        - {split_name:10s}: {stats['questions_with_useful']:,} useful queries "
                f"({stats['total_questions']:,} total across {stats['months']} months)"
            )

    elapsed = time.perf_counter() - t_start
    summary = {
        "elapsed_seconds": round(elapsed, 2),
        "cluster_tags_count": len(cluster_tags),
        "kept_questions": len(kept_qids),
        "score_sanity": sanity,
        "answer_capture_by_day": ans_cap,
        "vote_capture_by_day": vote_cap,
        "label_relevance_distribution": volumes["label_relevance_distribution"],
        "useful_answerers_per_query": volumes["useful_answerers_per_query"],
        "splits": splits_summary,
    }
    (tables_dir / "milestone1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done in {elapsed:.2f}s. Reports written to {cfg.paths.reports_dir}.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Use exact seed_tags without co-occurrence cluster expansion.",
    )
    parser.add_argument(
        "--skip-post-history",
        action="store_true",
        help="Skip PostHistory extraction (useful for quick smoke tests).",
    )
    args = parser.parse_args()
    run_milestone_1(
        config_path=args.config,
        expand_cluster=not args.no_expand,
        skip_post_history=args.skip_post_history,
    )


if __name__ == "__main__":
    main()
