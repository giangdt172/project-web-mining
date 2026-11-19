"""Fixed-horizon label construction, horizon curves, and monthly volume checks (Sections 1.1, 2.4)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from so_expert_rec.config import SplitsConfig


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_horizon_curves(
    questions_parquet: Path,
    answers_parquet: Path,
    votes_parquet: Path,
    figures_dir: Path | None = None,
    tables_dir: Path | None = None,
    reference_start: str | None = None,
    reference_end: str | None = None,
    max_answer_day: int = 60,
    max_vote_day: int = 120,
) -> dict[str, object]:
    """Compute empirical CDFs of answer delay (days since question) and vote delay (days since answer).

    Per Section 1.1, verifies whether H_a = 7 days and H_v = 30 days capture >= 85-90% of
    answers and votes respectively.
    """
    con = duckdb.connect()
    try:
        date_filter_sql = ""
        params: list[object] = [str(questions_parquet), str(answers_parquet)]
        if reference_start and reference_end:
            date_filter_sql = (
                " AND q.CreationDate >= CAST(? AS TIMESTAMP)"
                " AND q.CreationDate < CAST(? AS TIMESTAMP) + INTERVAL 1 DAY"
            )
            params.extend([reference_start, reference_end])

        answer_rows = con.execute(
            f"""
            WITH eligible AS (
                SELECT
                    GREATEST(
                        0,
                        CAST(FLOOR(DATE_DIFF('second', q.CreationDate, a.CreationDate) / 86400.0) AS BIGINT)
                    ) AS delay_day
                FROM read_parquet(?) q
                JOIN read_parquet(?) a ON a.ParentId = q.Id
                WHERE a.OwnerUserId IS NOT NULL
                  AND a.CommunityOwnedDate IS NULL
                  AND (q.OwnerUserId IS NULL OR a.OwnerUserId != q.OwnerUserId)
                  AND a.CreationDate >= q.CreationDate
                  {date_filter_sql}
            ),
            tot AS (SELECT COUNT(*)::DOUBLE AS n FROM eligible)
            SELECT
                LEAST(delay_day, {int(max_answer_day) + 1}) AS bucket_day,
                COUNT(*)::BIGINT AS cnt,
                (SELECT n FROM tot) AS total_n
            FROM eligible
            GROUP BY 1
            ORDER BY 1
            """,
            params,
        ).fetchall()

        vote_params: list[object] = [str(questions_parquet), str(answers_parquet), str(votes_parquet)]
        if reference_start and reference_end:
            vote_params.extend([reference_start, reference_end])

        vote_rows = con.execute(
            f"""
            WITH eligible_answers AS (
                SELECT a.Id AS answer_id, CAST(a.CreationDate AS DATE) AS ans_date
                FROM read_parquet(?) q
                JOIN read_parquet(?) a ON a.ParentId = q.Id
                WHERE a.OwnerUserId IS NOT NULL
                  AND a.CommunityOwnedDate IS NULL
                  AND (q.OwnerUserId IS NULL OR a.OwnerUserId != q.OwnerUserId)
                  AND a.CreationDate >= q.CreationDate
                  {date_filter_sql}
            ),
            vote_delays AS (
                SELECT
                    GREATEST(0, DATE_DIFF('day', ea.ans_date, CAST(v.CreationDate AS DATE))) AS delay_day
                FROM eligible_answers ea
                JOIN read_parquet(?) v ON v.PostId = ea.answer_id
                WHERE v.VoteTypeId IN (1, 2, 3)
            ),
            tot AS (SELECT COUNT(*)::DOUBLE AS n FROM vote_delays)
            SELECT
                LEAST(delay_day, {int(max_vote_day) + 1}) AS bucket_day,
                COUNT(*)::BIGINT AS cnt,
                (SELECT n FROM tot) AS total_n
            FROM vote_delays
            GROUP BY 1
            ORDER BY 1
            """,
            vote_params,
        ).fetchall()
    finally:
        con.close()

    def _build_cdf(rows: list[tuple[int, int, float]], max_day: int) -> tuple[list[dict[str, object]], dict[int, float]]:
        total = int(rows[0][2]) if rows and rows[0][2] else 0
        counts_by_day = {int(day): int(cnt) for day, cnt, _ in rows}
        curve: list[dict[str, object]] = []
        cum = 0
        for d in range(0, max_day + 1):
            cum += counts_by_day.get(d, 0)
            share = (cum / total) if total > 0 else 0.0
            curve.append({"delay_day": d, "count": counts_by_day.get(d, 0), "cum_count": cum, "cum_share": round(share, 6)})
        lookup = {int(r["delay_day"]): float(r["cum_share"]) for r in curve}
        return curve, lookup

    answer_curve, answer_cdf = _build_cdf(answer_rows, max_answer_day)
    vote_curve, vote_cdf = _build_cdf(vote_rows, max_vote_day)

    answer_milestones = {d: answer_cdf.get(d, 0.0) for d in (1, 3, 7, 14, 30) if d <= max_answer_day}
    vote_milestones = {d: vote_cdf.get(d, 0.0) for d in (7, 14, 30, 60, 90) if d <= max_vote_day}

    result: dict[str, object] = {
        "total_eligible_answers": int(answer_rows[0][2]) if answer_rows else 0,
        "total_answer_votes": int(vote_rows[0][2]) if vote_rows else 0,
        "answer_capture_by_day": answer_milestones,
        "vote_capture_by_day": vote_milestones,
        "answer_curve": answer_curve,
        "vote_curve": vote_curve,
    }

    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        _write_csv(tables_dir / "horizon_answer_curve.csv", ["delay_day", "count", "cum_count", "cum_share"], answer_curve)
        _write_csv(tables_dir / "horizon_vote_curve.csv", ["delay_day", "count", "cum_count", "cum_share"], vote_curve)

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax0, ax1 = axes

        ax0.plot([r["delay_day"] for r in answer_curve], [r["cum_share"] for r in answer_curve], lw=2)
        ax0.axvline(7, color="tab:red", linestyle="--", label="H_a = 7d")
        ax0.axhline(0.85, color="gray", linestyle=":", alpha=0.8)
        ax0.set_title("Answer Delay CDF (since question)")
        ax0.set_xlabel("Days since question")
        ax0.set_ylabel("Cumulative share of answers")
        ax0.set_ylim(0, 1.05)
        ax0.legend()

        ax1.plot([r["delay_day"] for r in vote_curve], [r["cum_share"] for r in vote_curve], lw=2, color="tab:green")
        ax1.axvline(30, color="tab:red", linestyle="--", label="H_v = 30d")
        ax1.axhline(0.85, color="gray", linestyle=":", alpha=0.8)
        ax1.set_title("Vote Delay CDF (since answer)")
        ax1.set_xlabel("Days since answer")
        ax1.set_ylabel("Cumulative share of votes")
        ax1.set_ylim(0, 1.05)
        ax1.legend()

        fig.tight_layout()
        fig.savefig(figures_dir / "horizon_curves.png", dpi=150)
        plt.close(fig)

    return result


def build_fixed_horizon_labels(
    questions_parquet: Path,
    answers_parquet: Path,
    votes_parquet: Path,
    answer_horizon_days: int = 7,
    vote_horizon_days: int = 30,
    output_path: Path | None = None,
    post_history_parquet: Path | None = None,
    useful_score_threshold: int = 1,
    dump_date: str | None = None,
) -> Path:
    """Build fixed-horizon graded relevance (0/1/2) and secondary 'responded' (0/1) labels.

    Section 1.1 rules:
    - Eligible answer: posted within `H_a` days of the question, non-self-answer,
      owned (`OwnerUserId IS NOT NULL`), and not community wiki.
    - Label score counts only upvotes (+1) and downvotes (-1) dated within `H_v` days of answer.
    - Graded relevance: accepted within `H_v` -> 2; else label_score >= threshold -> 1; else 0.
    - Secondary label `responded`: 1 for every eligible answer.
    - Last usable evaluation question date: `dump_date - H_a - H_v`.
    """
    out_path = Path(output_path) if output_path is not None else Path(questions_parquet).parent / "labels.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        if dump_date is None:
            max_dates = con.execute(
                """
                SELECT GREATEST(
                    COALESCE((SELECT MAX(CAST(CreationDate AS DATE)) FROM read_parquet(?)), DATE '1970-01-01'),
                    COALESCE((SELECT MAX(CAST(CreationDate AS DATE)) FROM read_parquet(?)), DATE '1970-01-01'),
                    COALESCE((SELECT MAX(CAST(CreationDate AS DATE)) FROM read_parquet(?)), DATE '1970-01-01')
                )
                """,
                [str(questions_parquet), str(answers_parquet), str(votes_parquet)],
            ).fetchone()
            inferred_dump_date = str(max_dates[0]) if max_dates and max_dates[0] else "2099-12-31"
        else:
            inferred_dump_date = dump_date

        if post_history_parquet is not None and Path(post_history_parquet).exists():
            dup_cte = f"""
            dup_closed AS (
                SELECT DISTINCT PostId AS question_id
                FROM read_parquet('{str(post_history_parquet)}')
                WHERE PostHistoryTypeId = 10
                  AND Comment IN ('1', '101')
            )
            """
        else:
            dup_cte = """
            dup_closed AS (
                SELECT CAST(NULL AS BIGINT) AS question_id WHERE FALSE
            )
            """

        h_a_secs = int(answer_horizon_days) * 86400
        h_v_days = int(vote_horizon_days)
        total_horizon_days = int(answer_horizon_days) + int(vote_horizon_days)

        query = f"""
        COPY (
            WITH {dup_cte},
            eligible_answers AS (
                SELECT
                    q.Id AS question_id,
                    a.Id AS answer_id,
                    a.OwnerUserId AS user_id,
                    q.OwnerUserId AS asker_id,
                    q.CreationDate AS question_creation_date,
                    a.CreationDate AS answer_creation_date,
                    CAST(a.CreationDate AS DATE) AS answer_date,
                    (dc.question_id IS NOT NULL) AS is_duplicate_closed,
                    (CAST(q.CreationDate AS DATE) <= CAST(? AS DATE) - INTERVAL {total_horizon_days} DAY)
                        AS within_usable_dump_horizon
                FROM read_parquet(?) q
                JOIN read_parquet(?) a ON a.ParentId = q.Id
                LEFT JOIN dup_closed dc ON dc.question_id = q.Id
                WHERE a.OwnerUserId IS NOT NULL
                  AND a.CommunityOwnedDate IS NULL
                  AND (q.OwnerUserId IS NULL OR a.OwnerUserId != q.OwnerUserId)
                  AND DATE_DIFF('second', q.CreationDate, a.CreationDate) >= 0
                  AND DATE_DIFF('second', q.CreationDate, a.CreationDate) <= {h_a_secs}
            ),
            horizon_votes AS (
                SELECT
                    ea.answer_id,
                    SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 WHEN v.VoteTypeId = 3 THEN -1 ELSE 0 END)::INTEGER
                        AS label_score,
                    MAX(CASE WHEN v.VoteTypeId = 1 THEN 1 ELSE 0 END)::INTEGER AS accepted_within_hv
                FROM eligible_answers ea
                JOIN read_parquet(?) v ON v.PostId = ea.answer_id
                WHERE v.VoteTypeId IN (1, 2, 3)
                  AND CAST(v.CreationDate AS DATE) >= ea.answer_date
                  AND CAST(v.CreationDate AS DATE) <= ea.answer_date + INTERVAL {h_v_days} DAY
                GROUP BY ea.answer_id
            )
            SELECT
                ea.question_id,
                ea.answer_id,
                ea.user_id,
                ea.asker_id,
                ea.question_creation_date,
                ea.answer_creation_date,
                COALESCE(hv.label_score, 0)::INTEGER AS label_score,
                COALESCE(hv.accepted_within_hv, 0)::INTEGER AS accepted_within_hv,
                CASE
                    WHEN COALESCE(hv.accepted_within_hv, 0) = 1 THEN 2
                    WHEN COALESCE(hv.label_score, 0) >= {int(useful_score_threshold)} THEN 1
                    ELSE 0
                END::INTEGER AS relevance,
                1::INTEGER AS responded,
                ea.is_duplicate_closed,
                ea.within_usable_dump_horizon
            FROM eligible_answers ea
            LEFT JOIN horizon_votes hv ON hv.answer_id = ea.answer_id
            ORDER BY ea.question_creation_date, ea.question_id, ea.answer_id
        ) TO '{str(out_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(
            query,
            [inferred_dump_date, str(questions_parquet), str(answers_parquet), str(votes_parquet)],
        )
    finally:
        con.close()

    return out_path


def check_monthly_volumes(
    questions_parquet: Path,
    labels_parquet: Path,
    splits: SplitsConfig | None = None,
    post_history_parquet: Path | None = None,
    figures_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> dict[str, object]:
    """Count monthly cluster questions with >= 1 useful (horizon) answer and summarize splits (Section 2.4)."""
    con = duckdb.connect()
    try:
        if post_history_parquet is not None and Path(post_history_parquet).exists():
            dup_cte = f"""
            dup_closed AS (
                SELECT DISTINCT PostId AS question_id
                FROM read_parquet('{str(post_history_parquet)}')
                WHERE PostHistoryTypeId = 10
                  AND Comment IN ('1', '101')
            )
            """
        else:
            dup_cte = """
            dup_closed AS (
                SELECT DISTINCT question_id
                FROM read_parquet(?)
                WHERE is_duplicate_closed
            )
            """

        has_ph = post_history_parquet is not None and Path(post_history_parquet).exists()
        monthly_params = (
            [str(labels_parquet), str(questions_parquet)]
            if has_ph
            else [str(labels_parquet), str(labels_parquet), str(questions_parquet)]
        )

        monthly_rows = con.execute(
            f"""
            WITH {dup_cte},
            q_labels AS (
                SELECT
                    question_id,
                    COUNT(DISTINCT user_id)::INTEGER AS responded_users,
                    COUNT(DISTINCT CASE WHEN relevance >= 1 THEN user_id END)::INTEGER AS useful_users,
                    COUNT(DISTINCT CASE WHEN relevance = 2 THEN user_id END)::INTEGER AS accepted_users
                FROM read_parquet(?)
                WHERE within_usable_dump_horizon
                GROUP BY question_id
            ),
            q_all AS (
                SELECT
                    q.Id AS question_id,
                    STRFTIME(q.CreationDate, '%Y-%m') AS month,
                    CAST(q.CreationDate AS DATE) AS q_date,
                    (dc.question_id IS NOT NULL) AS is_duplicate_closed,
                    COALESCE(ql.responded_users, 0) AS responded_users,
                    COALESCE(ql.useful_users, 0) AS useful_users,
                    COALESCE(ql.accepted_users, 0) AS accepted_users
                FROM read_parquet(?) q
                LEFT JOIN dup_closed dc ON dc.question_id = q.Id
                LEFT JOIN q_labels ql ON ql.question_id = q.Id
            )
            SELECT
                month,
                COUNT(*)::BIGINT AS total_questions,
                SUM(CASE WHEN NOT is_duplicate_closed THEN 1 ELSE 0 END)::BIGINT AS non_duplicate_questions,
                SUM(CASE WHEN NOT is_duplicate_closed AND responded_users >= 1 THEN 1 ELSE 0 END)::BIGINT
                    AS questions_with_responded,
                SUM(CASE WHEN NOT is_duplicate_closed AND useful_users >= 1 THEN 1 ELSE 0 END)::BIGINT
                    AS questions_with_useful,
                SUM(CASE WHEN NOT is_duplicate_closed AND accepted_users >= 1 THEN 1 ELSE 0 END)::BIGINT
                    AS questions_with_accepted,
                SUM(CASE WHEN NOT is_duplicate_closed AND useful_users >= 2 THEN 1 ELSE 0 END)::BIGINT
                    AS questions_with_2plus_useful
            FROM q_all
            GROUP BY month
            ORDER BY month
            """,
            monthly_params,
        ).fetchall()

        label_dist_rows = con.execute(
            """
            SELECT relevance, COUNT(*)::BIGINT AS cnt
            FROM read_parquet(?)
            WHERE NOT is_duplicate_closed AND within_usable_dump_horizon
            GROUP BY relevance
            ORDER BY relevance
            """,
            [str(labels_parquet)],
        ).fetchall()

        useful_mult_rows = con.execute(
            """
            WITH per_q AS (
                SELECT
                    question_id,
                    COUNT(DISTINCT CASE WHEN relevance >= 1 THEN user_id END)::INTEGER AS useful_users
                FROM read_parquet(?)
                WHERE NOT is_duplicate_closed AND within_usable_dump_horizon
                GROUP BY question_id
            )
            SELECT
                CASE WHEN useful_users >= 3 THEN '3+' ELSE CAST(useful_users AS VARCHAR) END AS bucket,
                COUNT(*)::BIGINT AS questions
            FROM per_q
            WHERE useful_users >= 1
            GROUP BY 1
            ORDER BY 1
            """,
            [str(labels_parquet)],
        ).fetchall()
    finally:
        con.close()

    monthly_records: list[dict[str, object]] = [
        {
            "month": str(r[0]),
            "total_questions": int(r[1]),
            "non_duplicate_questions": int(r[2]),
            "questions_with_responded": int(r[3]),
            "questions_with_useful": int(r[4]),
            "questions_with_accepted": int(r[5]),
            "questions_with_2plus_useful": int(r[6]),
        }
        for r in monthly_rows
    ]

    split_summary: dict[str, dict[str, int]] = {}
    if splits is not None:
        windows = {
            "history": (splits.history_start[:7], splits.history_end[:7]),
            "val_tune": (splits.val_tune_start[:7], splits.val_tune_end[:7]),
            "val_check": (splits.val_check_start[:7], splits.val_check_end[:7]),
            "val_total": (splits.val_tune_start[:7], splits.val_check_end[:7]),
            "test_main": (splits.test_main_start[:7], splits.test_main_end[:7]),
            "test_shift": (splits.test_shift_start[:7], splits.test_shift_end[:7]),
        }
        for name, (m_start, m_end) in windows.items():
            in_win = [r for r in monthly_records if m_start <= str(r["month"]) <= m_end]
            split_summary[name] = {
                "months": len(in_win),
                "total_questions": sum(int(r["total_questions"]) for r in in_win),
                "questions_with_useful": sum(int(r["questions_with_useful"]) for r in in_win),
                "questions_with_2plus_useful": sum(int(r["questions_with_2plus_useful"]) for r in in_win),
            }

    summary: dict[str, object] = {
        "total_months": len(monthly_records),
        "monthly": monthly_records,
        "splits": split_summary,
        "label_relevance_distribution": {int(r[0]): int(r[1]) for r in label_dist_rows},
        "useful_answerers_per_query": {str(r[0]): int(r[1]) for r in useful_mult_rows},
    }

    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        _write_csv(
            tables_dir / "monthly_volumes.csv",
            [
                "month",
                "total_questions",
                "non_duplicate_questions",
                "questions_with_responded",
                "questions_with_useful",
                "questions_with_accepted",
                "questions_with_2plus_useful",
            ],
            monthly_records,
        )
        (tables_dir / "volume_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if figures_dir is not None and monthly_records:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        months = [str(r["month"]) for r in monthly_records]
        totals = [int(r["total_questions"]) for r in monthly_records]
        useful = [int(r["questions_with_useful"]) for r in monthly_records]

        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(range(len(months)), totals, label="All cluster questions", lw=1.8)
        ax.plot(range(len(months)), useful, label="Queries with >=1 useful (horizon) answer", lw=2.0)
        step = max(1, len(months) // 12)
        ax.set_xticks(range(0, len(months), step))
        ax.set_xticklabels([months[i] for i in range(0, len(months), step)], rotation=45, ha="right")
        ax.set_ylabel("Questions / month")
        ax.set_title("Monthly Cluster Question & Evaluable Query Volume")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "monthly_volumes.png", dpi=150)
        plt.close(fig)

    return summary
