#!/usr/bin/env python3
"""Milestone 2: Staged tuning on validation months 1–4, overfitting check on months 5–6, and H1 (B3 vs B2)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from so_expert_rec.config import config_to_dict, load_config
from so_expert_rec.evaluation.tuning import run_overfitting_check, run_staged_tuning
from so_expert_rec.features.snapshot import PrecomputedStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--grid", type=Path, default=Path("configs/tuning_grid.yaml"))
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Evaluate default config on validation without grid search.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    tables_dir = cfg.paths.reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    store = PrecomputedStore.load(cfg.paths.processed_dir)

    if args.skip_tuning:
        tuned_cfg = cfg
        print("[1/2] Skipping grid search (--skip-tuning); using config defaults.")
    else:
        print("[1/2] Running Milestone 2 staged tuning (Stages 0, 1, 1b) on validation months 1–4...")
        tuned_cfg, tuning_report = run_staged_tuning(
            base_config=cfg,
            grid_path=args.grid,
            snapshots_dir=cfg.paths.snapshots_dir,
            output_dir=tables_dir,
            store=store,
        )
        print("      Stage 0 (candidate pool window W):")
        for row in tuning_report["stage_0_candidate_pool"]:
            print(
                f"        W={str(row['activity_window_days']):>4s}d | "
                f"mean_pool={row['mean_pool_size']:7.1f} | "
                f"cand_recall={row['candidate_recall']:.2%} | "
                f"cold_start={row['cold_start_share']:.2%}"
            )
        b_info = tuning_report["stage_1_best_vs_default_bootstrap"]
        print(
            f"      Stage 1 (B3): default NDCG@10={tuning_report['stage_1_default_b3_ndcg10']:.4f}, "
            f"best={tuning_report['stage_1_best_b3_ndcg10']:.4f}, "
            f"delta={b_info['delta']:+.4f} [{b_info['ci_low']:+.4f}, {b_info['ci_high']:+.4f}] -> "
            f"{'ADOPTED tuned' if b_info['adopted_tuned_over_default'] else 'KEPT default (parsimony)'}"
        )
        print(
            f"      Selected: W={tuned_cfg.candidates.activity_window_days}, "
            f"eps={tuned_cfg.quality.epsilon}, beta={tuned_cfg.quality.beta}, "
            f"tau={tuned_cfg.quality.tau_days}, m={tuned_cfg.quality.smoothing_m}, "
            f"variant={tuned_cfg.quality.expertise_variant} | "
            f"B0_W={tuned_cfg.baselines.b0_window_days}, B2_tau={tuned_cfg.baselines.b2_tau_days}"
        )

        tuned_yaml_path = tables_dir / "tuned_milestone2_config.yaml"
        tuned_yaml_path.write_text(yaml.safe_dump(config_to_dict(tuned_cfg), sort_keys=False), encoding="utf-8")

    print("[2/2] Running validation evaluation (months 5–6 check + full val + H1 + responded-vs-useful)...")
    val_report = run_overfitting_check(
        tuned_config=tuned_cfg,
        snapshots_dir=cfg.paths.snapshots_dir,
        output_dir=tables_dir,
        store=store,
    )

    for section_key, label in [
        ("val_check_months_5_6", "Validation Months 5–6 (Overfitting Check)"),
        ("val_all_months_1_6", "Validation Months 1–6 (Full Validation, Same-Tau H1)"),
    ]:
        sec = val_report[section_key]
        pool = sec["pool"]
        print(
            f"\n=== {label} ===\n"
            f"Queries: {pool['num_queries']} | Mean pool: {pool['mean_pool_size']:.1f} | "
            f"Candidate recall: {pool['candidate_recall']:.2%} | Cold-start share: {pool['cold_start_share']:.2%}"
        )
        print(f"{'Sys':<5s} {'NDCG@10':>8s} {'Rec@5':>8s} {'Rec@10':>8s} {'MRR@10':>8s} {'Top1%':>8s} {'Gini':>8s}")
        for sys_name, m in sec["systems"].items():
            print(
                f"{sys_name:<5s} {m['ndcg_at_10']:8.4f} {m['recall_at_5']:8.4f} "
                f"{m['recall_at_10']:8.4f} {m['mrr_at_10']:8.4f} "
                f"{m['top1pct_slot_share']:8.3f} {m['gini_top5']:8.3f}"
            )
        h1 = sec["h1_b3_vs_b2_same_tau"]
        print(
            f"H1 (B3 vs B2 same tau): delta={h1['delta_mean']:+.4f} "
            f"95% CI [{h1['ci_low']:+.4f}, {h1['ci_high']:+.4f}], "
            f"p={h1['raw_p_value']:.4g} ({'SIGNIFICANT' if h1['significant'] else 'not significant'})"
        )

    resp = val_report["responded_vs_useful_val_all"]
    print("\n=== Secondary Label ('responded') on Full Validation (Section 9.5) ===")
    for sys_name, m in resp["systems_responded"].items():
        print(f"  {sys_name}: NDCG@10={m['ndcg_at_10']:.4f}, Rec@10={m['recall_at_10']:.4f}, MRR@10={m['mrr_at_10']:.4f}")
    h1_r = resp["h1_responded_b3_vs_b2_same_tau"]
    print(
        f"  Responded H1 (B3 vs B2): delta={h1_r['delta_mean']:+.4f} "
        f"95% CI [{h1_r['ci_low']:+.4f}, {h1_r['ci_high']:+.4f}], p={h1_r['raw_p_value']:.4g}"
    )
    print(f"\nMilestone 2 completed in {time.perf_counter() - t0:.2f}s -> {tables_dir}")


if __name__ == "__main__":
    main()
