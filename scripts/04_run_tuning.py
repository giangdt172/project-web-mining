#!/usr/bin/env python3
"""Milestones 2–4: Staged validation tuning (Stages 0, 1, 1b, 2, 3-BM25), overfitting check, H1–H4 & latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from so_expert_rec.config import config_to_dict, load_config
from so_expert_rec.evaluation.tuning import (
    run_milestone3_tuning,
    run_milestone3_validation,
    run_milestone4_tuning,
    run_milestone4_validation,
    run_overfitting_check,
    run_staged_tuning,
)
from so_expert_rec.features.snapshot import PrecomputedStore


def _print_systems_table(sec: dict, label: str) -> None:
    pool = sec["pool"]
    print(
        f"\n=== {label} ===\n"
        f"Queries: {pool['num_queries']} | Mean pool: {pool['mean_pool_size']:.1f} | "
        f"Candidate recall: {pool['candidate_recall']:.2%} | Cold-start share: {pool['cold_start_share']:.2%}"
    )
    print(
        f"{'Sys':<12s} {'NDCG@10':>8s} {'Rec@5':>8s} {'Rec@10':>8s} "
        f"{'MRR@10':>8s} {'Top1%':>8s} {'Gini':>8s} {'Batch_p50':>10s}"
    )
    for sys_name, m in sec["systems"].items():
        lat_p50 = m.get("batch_latency_ms", {}).get("p50_ms", 0.0)
        print(
            f"{sys_name:<12s} {m['ndcg_at_10']:8.4f} {m['recall_at_5']:8.4f} "
            f"{m['recall_at_10']:8.4f} {m['mrr_at_10']:8.4f} "
            f"{m['top1pct_slot_share']:8.3f} {m['gini_top5']:8.3f} {lat_p50:8.1f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--grid", type=Path, default=Path("configs/tuning_grid.yaml"))
    parser.add_argument(
        "--milestone",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="Milestone to run: 2 (B0–B3, H1), 3 (B4/B4w/B5, H1–H3), or 4 (BM25 B6, H1–H4).",
    )
    parser.add_argument(
        "--stage2-from",
        type=Path,
        default=None,
        help="Optional tuned Milestone 2 YAML to start Stage 2 from without re-running Stages 0/1/1b.",
    )
    parser.add_argument(
        "--stage3-from",
        type=Path,
        default=None,
        help="Optional tuned Milestone 3 YAML to start Stage 3 (BM25 B6) from directly.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Evaluate given config on validation without grid search.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    start_cfg_path = args.stage3_from or args.stage2_from or args.config
    cfg = load_config(start_cfg_path)
    tables_dir = cfg.paths.reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    store = PrecomputedStore.load(cfg.paths.processed_dir)

    if args.milestone == 2:
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
                    f"        W={row['activity_window_days']!s:>4s}d | "
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
            (tables_dir / "tuned_milestone2_config.yaml").write_text(
                yaml.safe_dump(config_to_dict(tuned_cfg), sort_keys=False), encoding="utf-8"
            )

        print("[2/2] Running Milestone 2 validation evaluation (months 5–6 check + full val + H1)...")
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
            _print_systems_table(sec, label)
            h1 = sec["h1_b3_vs_b2_same_tau"]
            print(
                f"H1 (B3 vs B2 same tau): delta={h1['delta_mean']:+.4f} "
                f"95% CI [{h1['ci_low']:+.4f}, {h1['ci_high']:+.4f}], "
                f"p={h1['raw_p_value']:.4g} ({'SIGNIFICANT' if h1['significant'] else 'not significant'})"
            )
        print(f"\nMilestone 2 completed in {time.perf_counter() - t0:.2f}s -> {tables_dir}")
        return

    # --- Milestone 3 (or prerequisite for Milestone 4) ---
    if args.stage3_from is not None:
        m3_cfg = cfg
        print(f"[M3] Loaded Milestone 3 tuned config from {args.stage3_from}")
    else:
        m2_cfg = cfg
        if not args.skip_tuning and args.stage2_from is None:
            print("[M2] Running Milestone 2 staged tuning (Stages 0, 1, 1b) on validation months 1–4...")
            m2_cfg, _ = run_staged_tuning(
                base_config=cfg,
                grid_path=args.grid,
                snapshots_dir=cfg.paths.snapshots_dir,
                output_dir=tables_dir,
                store=store,
            )
            (tables_dir / "tuned_milestone2_config.yaml").write_text(
                yaml.safe_dump(config_to_dict(m2_cfg), sort_keys=False), encoding="utf-8"
            )
        elif args.stage2_from is not None:
            print(f"[M2] Loaded Milestone 2 tuned config from {args.stage2_from}")

        if args.skip_tuning:
            m3_cfg = m2_cfg
            print("[M3] Skipping Milestone 3 Stage 2 & B5 grid search (--skip-tuning).")
        else:
            print("[M3] Running Milestone 3 Stage 2 (B4w/B4) & B5 fusion tuning on validation months 1–4...")
            m3_cfg, m3_tune = run_milestone3_tuning(
                m2_config=m2_cfg,
                grid_path=args.grid,
                snapshots_dir=m2_cfg.paths.snapshots_dir,
                output_dir=tables_dir,
                store=store,
            )
            b4w_b = m3_tune["stage_2_b4w_best_vs_default_bootstrap"]
            print(
                f"      Stage 2 (B4w): default={m3_tune['stage_2_b4w_default_ndcg10']:.4f}, "
                f"best={m3_tune['stage_2_b4w_best_ndcg10']:.4f} {m3_tune['stage_2_b4w_best_params']}, "
                f"delta={b4w_b['delta']:+.4f} [{b4w_b['ci_low']:+.4f}, {b4w_b['ci_high']:+.4f}] -> "
                f"{'ADOPTED tuned' if b4w_b['adopted_tuned_over_default'] else 'KEPT default (parsimony)'}"
            )
            b5_b = m3_tune["stage_2_b5_linear_vs_rrf_bootstrap"]
            print(
                f"      Stage 2 (B5):  RRF={m3_tune['stage_2_b5_rrf_ndcg10']:.4f}, "
                f"best linear={m3_tune['stage_2_b5_best_linear_ndcg10']:.4f} "
                f"{m3_tune['stage_2_b5_best_linear_weights']}, "
                f"delta={b5_b['delta']:+.4f} [{b5_b['ci_low']:+.4f}, {b5_b['ci_high']:+.4f}] -> "
                f"{'ADOPTED linear' if b5_b['adopted_linear_over_rrf'] else 'KEPT RRF (parsimony)'}"
            )
            (tables_dir / "tuned_milestone3_config.yaml").write_text(
                yaml.safe_dump(config_to_dict(m3_cfg), sort_keys=False), encoding="utf-8"
            )

    if args.milestone == 3:
        print("[M3] Running Milestone 3 validation (B0–B5 + B4w_1hop, H1–H3 Holm, online PPR latency)...")
        val_report = run_milestone3_validation(
            tuned_config=m3_cfg,
            snapshots_dir=m3_cfg.paths.snapshots_dir,
            output_dir=tables_dir,
            store=store,
        )
        for section_key, label in [
            ("val_check_months_5_6", "Validation Months 5–6 (Overfitting Check)"),
            ("val_all_months_1_6", "Validation Months 1–6 (Full Validation, Strict H1–H3)"),
        ]:
            sec = val_report[section_key]
            _print_systems_table(sec, label)
            print("Hypotheses (paired bootstrap 95% CI, paired randomization, Holm-Bonferroni):")
            for hid in ("H1", "H2", "H3"):
                h = sec["hypotheses_holm"][hid]
                print(
                    f"  {hid} ({h['system_a']} vs {h['system_b']}): "
                    f"delta={h['delta_mean']:+.4f} [{h['ci_low']:+.4f}, {h['ci_high']:+.4f}], "
                    f"raw_p={h['raw_p_value']:.4g}, holm_p={h['holm_p_value']:.4g} "
                    f"({'SIGNIFICANT' if h['significant'] else 'not significant'})"
                )
            oh = sec["one_hop_diagnostic_b4w_vs_1hop"]
            print(
                f"  One-hop diagnostic (B4w vs B4w_1hop): "
                f"delta={oh['delta_mean']:+.4f} [{oh['ci_low']:+.4f}, {oh['ci_high']:+.4f}], "
                f"p={oh['raw_p_value']:.4g}"
            )
        lat = val_report.get("online_ppr_latency_benchmark", {})
        if lat:
            ol = lat["online_per_query_latency_ms"]
            print(
                f"\n=== Demo / Online Per-Tag Cached PPR Latency ({lat['month']}, N={lat['num_queries']}) ===\n"
                f"  Online demo path: p50={ol['p50_ms']:.3f} ms, p95={ol['p95_ms']:.3f} ms | "
                f"max diff={lat['max_abs_diff_online_vs_batched']:.2e}"
            )
        print(f"\nMilestone 3 completed in {time.perf_counter() - t0:.2f}s -> {tables_dir}")
        return

    # --- Milestone 4 (BM25 + B6 + H4) ---
    b5_fusion_ref = (m3_cfg.fusion.method, dict(m3_cfg.fusion.weights))
    if args.skip_tuning:
        m4_cfg = m3_cfg
        print("[M4] Skipping Stage 3 BM25 grid search (--skip-tuning).")
    else:
        print("[M4] Running Milestone 4 Stage 3 (BM25 K, gamma, 3-way fusion for B6) on val months 1–4...")
        m4_cfg, m4_tune = run_milestone4_tuning(
            m3_config=m3_cfg,
            grid_path=args.grid,
            snapshots_dir=m3_cfg.paths.snapshots_dir,
            output_dir=tables_dir,
            store=store,
        )
        lr = m4_tune["stage_3_bm25_linear_vs_rrf_bootstrap"]
        wd = m4_tune["stage_3_bm25_winner_vs_default_bootstrap"]
        print(
            f"      Stage 3 (BM25): default(K=50,g=0.4,lin 0.5/0.3/0.2)={m4_tune['stage_3_bm25_default_ndcg10']:.4f}, "
            f"best RRF={m4_tune['stage_3_bm25_best_rrf_ndcg10']:.4f} {m4_tune['stage_3_bm25_best_rrf_params']}, "
            f"best linear={m4_tune['stage_3_bm25_best_linear_ndcg10']:.4f} {m4_tune['stage_3_bm25_best_linear_params']}"
        )
        print(
            f"      Linear vs RRF: delta={lr['delta']:+.4f} [{lr['ci_low']:+.4f}, {lr['ci_high']:+.4f}] -> "
            f"{'ADOPTED linear' if lr['adopted_linear_over_rrf'] else 'KEPT RRF (parsimony)'}"
        )
        print(
            f"      Winner vs Default: delta={wd['delta']:+.4f} [{wd['ci_low']:+.4f}, {wd['ci_high']:+.4f}] -> "
            f"{'ADOPTED tuned' if wd['adopted_tuned_over_default'] else 'KEPT default (parsimony)'}"
        )
        print(
            f"      Selected M4: variant={m4_cfg.retrieval.variant}, K={m4_cfg.retrieval.top_k}, "
            f"gamma={m4_cfg.retrieval.seed_gamma}, fusion={m4_cfg.fusion.method} ({m4_cfg.fusion.weights})"
        )
        (tables_dir / "tuned_milestone4_config.yaml").write_text(
            yaml.safe_dump(config_to_dict(m4_cfg), sort_keys=False), encoding="utf-8"
        )

    print("[M4] Running Milestone 4 validation (B0–B6, H1–H4 Holm, online B6 latency)...")
    val_report = run_milestone4_validation(
        m4_config=m4_cfg,
        m3_b5_fusion=b5_fusion_ref,
        snapshots_dir=m4_cfg.paths.snapshots_dir,
        output_dir=tables_dir,
        store=store,
    )

    for section_key, label in [
        ("val_check_months_5_6", "Validation Months 5–6 (Overfitting Check)"),
        ("val_all_months_1_6", "Validation Months 1–6 (Full Validation, Strict H1–H4)"),
    ]:
        sec = val_report[section_key]
        _print_systems_table(sec, label)
        print("Hypotheses (paired bootstrap 95% CI, paired randomization, Holm-Bonferroni across H1–H4):")
        for hid in ("H1", "H2", "H3", "H4"):
            h = sec["hypotheses_holm"][hid]
            print(
                f"  {hid} ({h['system_a']} vs {h['system_b']}): "
                f"delta={h['delta_mean']:+.4f} [{h['ci_low']:+.4f}, {h['ci_high']:+.4f}], "
                f"raw_p={h['raw_p_value']:.4g}, holm_p={h['holm_p_value']:.4g} "
                f"({'SIGNIFICANT' if h['significant'] else 'not significant'})"
            )
        d1 = sec["diagnostic_b6_vs_b3"]
        d2 = sec["diagnostic_ppr_text_vs_b4w"]
        print(
            f"  Decision-rule check (B6 vs B3): delta={d1['delta_mean']:+.4f} "
            f"[{d1['ci_low']:+.4f}, {d1['ci_high']:+.4f}], p={d1['raw_p_value']:.4g}"
        )
        print(
            f"  Seed diagnostic (B6_ppr_text vs B4w): delta={d2['delta_mean']:+.4f} "
            f"[{d2['ci_low']:+.4f}, {d2['ci_high']:+.4f}], p={d2['raw_p_value']:.4g}"
        )

    lat = val_report.get("online_b6_latency_benchmark", {})
    if lat:
        lb = lat["online_bm25_ms"]
        lp = lat["online_ppr_with_qseeds_ms"]
        lt = lat["online_b6_total_ms"]
        print(
            f"\n=== Demo / Online B6 Latency ({lat['month']}, N={lat['num_queries']}, K={lat['top_k']}, gamma={lat['seed_gamma']}) ===\n"
            f"  BM25 retrieval:    p50={lb['p50_ms']:.3f} ms, p95={lb['p95_ms']:.3f} ms/query\n"
            f"  Online PPR+qseeds: p50={lp['p50_ms']:.3f} ms, p95={lp['p95_ms']:.3f} ms/query\n"
            f"  Total online B6:   p50={lt['p50_ms']:.3f} ms, p95={lt['p95_ms']:.3f} ms/query\n"
            f"  Max |online - batched| question-seeded PPR diff: {lat['max_abs_diff_online_vs_batched_ppr']:.2e}"
        )

    print(f"\nMilestone 4 completed in {time.perf_counter() - t0:.2f}s -> {tables_dir}")


if __name__ == "__main__":
    main()
