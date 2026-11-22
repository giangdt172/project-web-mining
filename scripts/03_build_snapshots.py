#!/usr/bin/env python3
"""Milestones 2–4, 6: Build and cache monthly point-in-time snapshots (q_tau, E, graphs, BM25)."""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

from so_expert_rec.config import load_config
from so_expert_rec.features.snapshot import (
    PrecomputedStore,
    build_or_load_bm25_snapshot,
    build_or_load_graph_snapshot,
    build_or_load_snapshot,
    load_question_bm25_texts,
)


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


def run_build_snapshots(
    config_path: Path,
    split: str = "all",
    rebuild: bool = False,
    include_graph: bool = True,
    include_bm25: bool = True,
) -> None:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    store = PrecomputedStore.load(cfg.paths.processed_dir)
    corpus_bm25_texts = load_question_bm25_texts(cfg.paths.processed_dir, store) if include_bm25 else None

    months: list[str] = []
    if split in ("val", "all"):
        months.extend(_months_between(cfg.splits.val_tune_start, cfg.splits.val_check_end))
    if split in ("test_main", "all"):
        months.extend(_months_between(cfg.splits.test_main_start, cfg.splits.test_main_end))
    if split in ("test_shift", "all"):
        months.extend(_months_between(cfg.splits.test_shift_start, cfg.splits.test_shift_end))
    months = sorted(set(months))

    print(
        f"Building/loading {len(months)} monthly snapshots ({split}, "
        f"graph={'on' if include_graph else 'off'}, bm25={'on' if include_bm25 else 'off'}) "
        f"into {cfg.paths.snapshots_dir}..."
    )
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
            build_or_load_snapshot(
                month=month,
                config=cfg,
                processed_dir=cfg.paths.processed_dir,
                snapshots_dir=cfg.paths.snapshots_dir,
                store=store,
                use_cache=True,
            )

        graph_info = ""
        if include_graph:
            w_graph = build_or_load_graph_snapshot(
                month=month,
                config=cfg,
                processed_dir=cfg.paths.processed_dir,
                snapshots_dir=cfg.paths.snapshots_dir,
                store=store,
                snapshot=snap,
                unweighted=False,
                use_cache=not rebuild,
            )
            u_graph = build_or_load_graph_snapshot(
                month=month,
                config=cfg,
                processed_dir=cfg.paths.processed_dir,
                snapshots_dir=cfg.paths.snapshots_dir,
                store=store,
                snapshot=snap,
                unweighted=True,
                use_cache=not rebuild,
            )
            if rebuild:
                build_or_load_graph_snapshot(
                    month=month,
                    config=cfg,
                    processed_dir=cfg.paths.processed_dir,
                    snapshots_dir=cfg.paths.snapshots_dir,
                    store=store,
                    snapshot=snap,
                    unweighted=False,
                    use_cache=True,
                )
                build_or_load_graph_snapshot(
                    month=month,
                    config=cfg,
                    processed_dir=cfg.paths.processed_dir,
                    snapshots_dir=cfg.paths.snapshots_dir,
                    store=store,
                    snapshot=snap,
                    unweighted=True,
                    use_cache=True,
                )
            graph_info = (
                f" | G_w_nnz={w_graph.p_transpose.nnz:,} (kappa={w_graph.kappa:.3f}) "
                f"G_u_nnz={u_graph.p_transpose.nnz:,}"
            )

        bm25_info = ""
        if include_bm25:
            bm25_idx = build_or_load_bm25_snapshot(
                month=month,
                processed_dir=cfg.paths.processed_dir,
                snapshots_dir=cfg.paths.snapshots_dir,
                store=store,
                snapshot=snap,
                question_bm25_texts=corpus_bm25_texts,
                use_cache=not rebuild,
            )
            if rebuild:
                build_or_load_bm25_snapshot(
                    month=month,
                    processed_dir=cfg.paths.processed_dir,
                    snapshots_dir=cfg.paths.snapshots_dir,
                    store=store,
                    snapshot=snap,
                    question_bm25_texts=corpus_bm25_texts,
                    use_cache=True,
                )
            bm25_info = f" | BM25_docs={bm25_idx.num_docs:,}"

        dt_ms = (time.perf_counter() - tm) * 1000.0
        print(
            f"  {month}: Q={snap.num_questions:,} A={snap.num_answers:,} "
            f"E_prod_nnz={snap.expertise.e_prod.nnz:,}{graph_info}{bm25_info} ({dt_ms:.1f} ms)"
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
    parser.add_argument("--no-graph", action="store_true", help="Skip building heterogeneous graph caches.")
    parser.add_argument("--no-bm25", action="store_true", help="Skip building BM25 snapshot indices.")
    args = parser.parse_args()
    run_build_snapshots(
        args.config,
        split=args.split,
        rebuild=args.rebuild,
        include_graph=not args.no_graph,
        include_bm25=not args.no_bm25,
    )


if __name__ == "__main__":
    main()
