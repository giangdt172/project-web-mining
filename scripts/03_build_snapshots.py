#!/usr/bin/env python3
"""Milestones 2–4, 6: Build and cache monthly point-in-time snapshots (q_tau, E_sum, n_tau, E_prod)."""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

from so_expert_rec.config import load_config
from so_expert_rec.features.snapshot import PrecomputedStore, build_or_load_snapshot


def _months_between(start_iso: str, end_iso: str) -> list[str]:
    s_y, s_m = int(start_iso[:4]), int(start_iso[5:7])
    e_y, e_m = int(end_iso[:4]), int(end_iso[5:7])
    months: list[str] = []
    cur = date(s_y, s_m, 1)
    end = date(e_y, e_m, 1)
    while cur <= end:
        months.append(f"{cur.year:04d}-{cur.month:02d}")
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def run_build_snapshots(config_path: Path, split: str = "all", rebuild: bool = False) -> None:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    store = PrecomputedStore.load(cfg.paths.processed_dir)

    months: list[str] = []
    if split in ("val", "all"):
        months.extend(_months_between(cfg.splits.val_tune_start, cfg.splits.val_check_end))
    if split in ("test_main", "all"):
        months.extend(_months_between(cfg.splits.test_main_start, cfg.splits.test_main_end))
    if split in ("test_shift", "all"):
        months.extend(_months_between(cfg.splits.test_shift_start, cfg.splits.test_shift_end))
    months = sorted(set(months))

    print(f"Building/loading {len(months)} monthly snapshots ({split}) into {cfg.paths.snapshots_dir}...")
    for month in months:
        tm = time.perf_counter()
        snap = build_or_load_snapshot(
            month=month,
            config=cfg,
            processed_dir=cfg.paths.processed_dir,
            snapshots_dir=cfg.paths.snapshots_dir,
            store=store,
            use_cache=not rebuild,
        )
        if rebuild:
            # Persist fresh cache after force rebuild
            build_or_load_snapshot(
                month=month,
                config=cfg,
                processed_dir=cfg.paths.processed_dir,
                snapshots_dir=cfg.paths.snapshots_dir,
                store=store,
                use_cache=True,
            )
        dt_ms = (time.perf_counter() - tm) * 1000.0
        print(
            f"  {month}: Q={snap.num_questions:,} A={snap.num_answers:,} "
            f"E_prod_nnz={snap.expertise.e_prod.nnz:,} ({dt_ms:.1f} ms)"
        )
    print(f"Snapshots ready in {time.perf_counter() - t0:.2f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--split",
        choices=["val", "test_main", "test_shift", "all"],
        default="val",
        help="Which split months to build snapshots for (default: val).",
    )
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing cache and rebuild.")
    args = parser.parse_args()
    run_build_snapshots(args.config, split=args.split, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
