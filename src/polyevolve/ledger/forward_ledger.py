"""Forward paper-bet ledger - the platform's un-foolable validator.

Belief is earned ONLY here: a candidate edge logs a forward prediction (our fair
P(YES) for a specific market) alongside the crowd price AT THE TIME, and we grade it
against reality AT RESOLUTION. Forward + out-of-sample can't be overfit or faked by a
mid-quote, so this is the one gate the rest of the platform funnels into
(see ARCHITECTURE.md).

A bet is scored on Brier-vs-fair: did our probability beat the crowd's on the actual
outcome? (A magnitude fade is CORRECT if the favorite wins but our prob was better
calibrated.) Markets with a `market_external_id` auto-grade via Polymarket; others are
graded manually (set --outcome) for real-world facts not tied to one clean binary.

CLI:
    uv run python -m polyevolve.forward_ledger init
    uv run python -m polyevolve.forward_ledger log --category politics --question "..." \
        --market-id 123 --crowd 0.735 --fair 0.63 --rule C --confidence low-med \
        --resolution-date 2026-06-18 --notes "..."
    uv run python -m polyevolve.forward_ledger grade        # auto-resolve open bets
    uv run python -m polyevolve.forward_ledger grade --id 7 --outcome NO   # manual
    uv run python -m polyevolve.forward_ledger report
"""

from __future__ import annotations

import argparse
import os

import httpx
import psycopg

from polyevolve.markets.polymarket import GAMMA_BASE, PolymarketMarket
from polyevolve.storage import db

DB_URL = os.environ.get("DB_URL", "postgresql://superpod:superpod@localhost:5432/superpod")

# DDL for the forward paper-bet ledger. Use ensure_table(cur) to apply it rather
# than referencing this string directly (the harness inserter calls that path).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_bets (
    id                  SERIAL PRIMARY KEY,
    logged_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    category            TEXT NOT NULL,
    venue               TEXT NOT NULL DEFAULT 'polymarket',
    market_external_id  TEXT,                 -- NULL => grade manually
    question            TEXT NOT NULL,
    crowd_price         DOUBLE PRECISION NOT NULL,   -- crowd P(YES) at log time
    fair_estimate       DOUBLE PRECISION NOT NULL,   -- our P(YES)
    rule                TEXT,                 -- which through-line/edge fired
    confidence          TEXT,
    resolution_date     DATE,
    notes               TEXT,
    status              TEXT NOT NULL DEFAULT 'open', -- open | resolved | void
    actual_outcome      TEXT,                 -- YES | NO
    brier_fair          DOUBLE PRECISION,
    brier_crowd         DOUBLE PRECISION,
    beat_crowd          BOOLEAN,
    resolved_at         TIMESTAMPTZ
);
"""


def _brier(p: float, outcome: str) -> float:
    y = 1.0 if outcome == "YES" else 0.0
    return (p - y) ** 2


def ensure_table(cur: psycopg.Cursor) -> None:
    """Create the `paper_bets` table if it does not exist (idempotent).

    The public way to provision the ledger schema. `init()` and the harness's
    ledger-insert (`polyevolve run`) both call this instead of reaching for the
    private DDL string, so there is one place that owns the table definition.
    """
    cur.execute(_SCHEMA)


def init() -> None:
    with db.connection(DB_URL) as conn, conn.cursor() as cur:
        ensure_table(cur)
        conn.commit()
    print("paper_bets table ready.")


def log_bet(args: argparse.Namespace) -> None:
    with db.connection(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO paper_bets (category, market_external_id, question, crowd_price,
                fair_estimate, rule, confidence, resolution_date, notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
            """,
            (
                args.category,
                args.market_id,
                args.question,
                args.crowd,
                args.fair,
                args.rule,
                args.confidence,
                args.resolution_date,
                args.notes,
            ),
        )
        row = cur.fetchone()
        bet_id = row[0] if row else None
        conn.commit()
    print(f"logged paper bet #{bet_id}: {args.question[:60]}  crowd={args.crowd} fair={args.fair}")


def _settle(cur: psycopg.Cursor, row_id: int, outcome: str, crowd: float, fair: float) -> None:
    bf, bc = _brier(fair, outcome), _brier(crowd, outcome)
    cur.execute(
        """
        UPDATE paper_bets SET status='resolved', actual_outcome=%s, brier_fair=%s,
            brier_crowd=%s, beat_crowd=%s, resolved_at=now() WHERE id=%s
        """,
        (outcome, bf, bc, bf < bc, row_id),
    )


def grade(args: argparse.Namespace) -> None:
    with db.connection(DB_URL) as conn, conn.cursor() as cur:
        # Manual single-bet grade.
        if args.id is not None and args.outcome is not None:
            cur.execute(
                "SELECT crowd_price, fair_estimate, question FROM paper_bets WHERE id=%s",
                (args.id,),
            )
            r = cur.fetchone()
            if not r:
                print(f"no bet #{args.id}")
                return
            _settle(cur, args.id, args.outcome, r[0], r[1])
            conn.commit()
            print(f"#{args.id} graded {args.outcome} ({r[2][:50]})")
            return
        # Auto-grade all open bets that have a market id.
        cur.execute(
            "SELECT id, market_external_id, crowd_price, fair_estimate, question "
            "FROM paper_bets WHERE status='open' AND market_external_id IS NOT NULL"
        )
        rows = cur.fetchall()
    src = PolymarketMarket(httpx.Client(base_url=GAMMA_BASE, timeout=20))
    graded = 0
    with db.connection(DB_URL) as conn, conn.cursor() as cur:
        for bet_id, mid, crowd, fair, q in rows:
            res = src.get_resolution(str(mid))
            if res is None:
                print(f"#{bet_id} still open: {q[:50]}")
                continue
            _settle(cur, bet_id, res.outcome, crowd, fair)
            graded += 1
            print(f"#{bet_id} RESOLVED {res.outcome}: {q[:50]}")
        conn.commit()
    print(f"\nauto-graded {graded} bet(s); {len(rows) - graded} still open.")


def report(args: argparse.Namespace) -> None:
    with db.connection(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT category, status, rule, question, crowd_price, fair_estimate, "
            "actual_outcome, brier_fair, brier_crowd, beat_crowd FROM paper_bets ORDER BY id"
        )
        rows = cur.fetchall()
    if not rows:
        print("no paper bets logged.")
        return
    resolved = [r for r in rows if r[1] == "resolved"]
    open_ = [r for r in rows if r[1] == "open"]
    print(
        f"\nFORWARD PAPER-BET LEDGER - {len(rows)} bets "
        f"({len(resolved)} resolved, {len(open_)} open)\n"
    )
    for r in rows:
        cat, st, rule, q, crowd, fair, out, bf, bc, beat = r
        tag = "OPEN " if st == "open" else (f"{out} " + ("WIN " if beat else "loss"))
        extra = f" brier fair={bf:.3f} crowd={bc:.3f}" if st == "resolved" else ""
        print(f"  [{tag:9}] {rule or '?':4} {q[:46]:46} crowd={crowd:.2f} fair={fair:.2f}{extra}")
    if resolved:
        n = len(resolved)
        wins = sum(1 for r in resolved if r[9])
        mbf = sum(r[7] for r in resolved) / n
        mbc = sum(r[8] for r in resolved) / n
        print(
            f"\n  RESOLVED n={n}: beat-crowd {wins}/{n}  "
            f"mean Brier fair={mbf:.3f} vs crowd={mbc:.3f}  (edge={mbc - mbf:+.3f})"
        )
        print(
            "  ⚠ FDR: a forward win-rate is only meaningful at n>=~20-30;"
            " below that, treat as anecdote."
        )
    else:
        print("\n  (no resolved bets yet - edge is UNCONFIRMED until reality grades it)")


def main() -> int:
    ap = argparse.ArgumentParser(description="forward paper-bet ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    lg = sub.add_parser("log")
    lg.add_argument("--category", required=True)
    lg.add_argument("--question", required=True)
    lg.add_argument("--market-id", dest="market_id", default=None)
    lg.add_argument("--crowd", type=float, required=True)
    lg.add_argument("--fair", type=float, required=True)
    lg.add_argument("--rule", default=None)
    lg.add_argument("--confidence", default=None)
    lg.add_argument("--resolution-date", dest="resolution_date", default=None)
    lg.add_argument("--notes", default=None)
    gr = sub.add_parser("grade")
    gr.add_argument("--id", type=int, default=None)
    gr.add_argument("--outcome", choices=["YES", "NO"], default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    {"init": lambda a: init(), "log": log_bet, "grade": grade, "report": report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
