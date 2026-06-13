"""Single entrypoint CLI for the market-experiment platform.

The three platform verbs (run as ``polyevolve <cmd>`` or ``python -m polyevolve.cli <cmd>``):
    polyevolve scout [categories...]              live category x thinness map
    polyevolve run --market polymarket \\          one experiment through the harness:
        --forecaster baseline --connectors news   pull -> research -> predict -> score
        --category politics --lead-days 30         -> rubric -> forward ledger
    polyevolve ledger report                       forward paper-bet ledger (init/log/grade/report)

Plus the existing DB inspection verbs (predictions/calibration/cost/runs/coverage/
traces/backtest/snapshots/evaluate/sweep). See CONTRIBUTING.md to add a plugin.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

import psycopg

from polyevolve.config import Config
from polyevolve.core.types import MarketFilter

if TYPE_CHECKING:
    from polyevolve.evolution.evaluator import EvalResult


def _render(cur: psycopg.Cursor, *, title: str = "") -> None:
    from polyevolve.cli_ui import color, table

    cols = [d.name for d in cur.description or []]
    rows = cur.fetchall()
    if not rows:
        print(color(f"  {title or 'query'}: (no rows)", "dim"))
        return
    trows = [[_fmt(v) for v in r] for r in rows]
    print(table(cols, trows, title=title))
    print(color(f"  ({len(rows)} rows)", "dim"))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _query(cfg: Config, sql: str, params: tuple[Any, ...] = (), *, title: str = "") -> None:
    with psycopg.connect(cfg.db_url) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        _render(cur, title=title)


def cmd_predictions(cfg: Config, args: argparse.Namespace) -> None:
    _query(
        cfg,
        "SELECT * FROM v_recent_predictions LIMIT %s",
        (args.limit,),
        title="recent predictions",
    )


def cmd_calibration(cfg: Config, args: argparse.Namespace) -> None:
    _query(cfg, "SELECT * FROM calibration", title="decile calibration (resolved markets only)")
    print()
    _query(cfg, "SELECT * FROM v_calibration_vs_market", title="calibration vs market")


def cmd_cost(cfg: Config, args: argparse.Namespace) -> None:
    from polyevolve.cli_ui import color

    _query(cfg, "SELECT * FROM v_cost", title="token usage + estimated cost")
    with psycopg.connect(cfg.db_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT ROUND(SUM(estimated_cost_usd), 4) FROM llm_calls")
        total = cur.fetchone()
        print(color(f"  total estimated cost: ${(total[0] if total and total[0] else 0)}", "bold"))


def cmd_runs(cfg: Config, args: argparse.Namespace) -> None:
    _query(cfg, "SELECT * FROM v_run_summary", title="per-day run summary")


def cmd_coverage(cfg: Config, args: argparse.Namespace) -> None:
    _query(cfg, "SELECT * FROM v_market_coverage", title="market coverage by status")


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> None:
    from polyevolve.cli_ui import color

    run = args.run
    if run is None:
        with psycopg.connect(cfg.db_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT run_id FROM backtests ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            run = row[0] if row else None
    if run is None:
        print(color("  (no backtest runs yet)", "dim"))
        return
    _query(
        cfg,
        "SELECT * FROM v_backtest_calibration WHERE run_id = %s",
        (run,),
        title=f"backtest run: {run}",
    )
    print(color("  trust ONLY the is_clean=true / holdout cohort for go/no-go.", "dim"))


def cmd_snapshot(cfg: Config, args: argparse.Namespace) -> None:
    """Build a frozen, resolved-market dataset (eval_snapshots) to evolve against."""
    from datetime import UTC, datetime

    from polyevolve.orchestration.snapshot import build_snapshot

    build_snapshot(
        cfg=cfg,
        now=datetime.now(UTC),
        snapshot_set=args.snapshot_set,
        limit=args.limit,
        lead_days=args.lead_days,
        max_per_event=args.max_per_event,
        domain=args.domain,
        min_volume=args.min_volume,
        gather_research=not args.no_research,
        pages_per_tag=args.pages_per_tag,
    )


def cmd_snapshots(cfg: Config, args: argparse.Namespace) -> None:
    _query(
        cfg,
        """
        SELECT snapshot_set,
               COUNT(*) AS n,
               COUNT(*) FILTER (WHERE market_price_at_as_of IS NOT NULL) AS priced,
               COUNT(*) FILTER (WHERE outcome = 'YES') AS yes,
               MIN(resolved_at)::date AS earliest,
               MAX(resolved_at)::date AS latest
        FROM eval_snapshots GROUP BY snapshot_set ORDER BY snapshot_set
        """,
        title="frozen eval snapshot sets",
    )


def _print_eval(label: str, result: EvalResult) -> None:
    from polyevolve.cli_ui import color, panel

    r = result

    def fmt(x: float | None) -> str:
        return f"{x:+.4f}" if x is not None else "n/a"

    note = color("(holdout is the trustworthy one)", "dim")
    print()
    print(
        panel(
            [
                f"n_total {r.n_total}   n_clean {r.n_clean}   priced_clean {r.n_priced_clean}",
                f"cache   {r.cache_hits} hits, {r.cache_misses} misses, {r.failed} failed",
                f"brier   train {fmt(r.brier_train)}   holdout {fmt(r.brier_holdout)}",
                f"edge    train {fmt(r.edge_train)}   holdout {fmt(r.edge_holdout)}",
                f"score   {fmt(r.combined_score)}   {note}",
            ],
            title=label,
        )
    )


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> None:
    from polyevolve.evolution.evaluator import evaluate
    from polyevolve.evolution.genome import default_genome
    from polyevolve.models import build_model

    model = build_model(model_id=args.model, anthropic_api_key=cfg.anthropic_api_key)
    result = evaluate(
        genome=default_genome(),
        model=model,
        db_url=cfg.db_url,
        snapshot_set=args.snapshot_set,
    )
    _print_eval(f"{model.name} on {args.snapshot_set}", result)


def cmd_sweep(cfg: Config, args: argparse.Namespace) -> None:
    from polyevolve.evolution.evaluator import evaluate
    from polyevolve.evolution.genome import default_genome
    from polyevolve.models import build_model

    genome = default_genome()
    for model_id in args.models:
        model = build_model(model_id=model_id, anthropic_api_key=cfg.anthropic_api_key)
        result = evaluate(
            genome=genome, model=model, db_url=cfg.db_url, snapshot_set=args.snapshot_set
        )
        _print_eval(f"{model.name} on {args.snapshot_set}", result)


def cmd_evolve(cfg: Config, args: argparse.Namespace) -> None:
    """Evolve a strategy genome and report seed -> champion on an honest holdout."""
    import polyevolve.api as pe

    qs = pe.markets(
        source=args.source,
        path=args.path,
        snapshot_set=args.snapshot_set,
        db_url=cfg.db_url,
        limit=args.limit,
    )
    if len(qs) < 8:
        print(
            f"need >=8 resolved markets to evolve; got {len(qs)}. Build some first, e.g.:\n"
            f"  polyevolve snapshot --set demo --domain all --min-volume 10000 "
            f"--no-research --limit 200"
        )
        return
    # Honest holdout: the last `holdout` fraction is validation, never selected on.
    cut = int(len(qs) * (1.0 - args.holdout))
    train, val = qs[:cut], qs[cut:]
    pools = pe.gather(train) if args.gather else None
    vpools = pe.gather(val) if args.gather else None

    from polyevolve.bench.scoring import brier as _brier
    from polyevolve.cli_ui import color, panel, spark

    metric = "-Brier (higher=better)" if args.objective == "calibration" else "net-of-spread ROI"
    knobs = pe.SeedKnobs()
    src = args.snapshot_set or args.path or args.source
    print()
    print(
        panel(
            [
                f"model      {knobs.model_id}",
                f"dataset    {src}  -  {len(train)} train / {len(val)} holdout",
                f"objective  {args.objective}  -  prompt-mutation: on",
                f"budget     {args.generations} generations x pop {args.pop}",
            ],
            title="polyevolve evolve",
        )
    )

    print(f"\n  {'gen':>5}  {'train':>9}  {'holdout':>9}")
    hist_va: list[float] = []

    def _progress(gen: int, gens: int, tr: float, va: float) -> None:
        hist_va.append(va)
        print(f"  {f'{gen}/{gens}':>5}  {tr:>+9.4f}  {va:>+9.4f}", flush=True)

    res = pe.evolve(
        train,
        pools,
        objective=args.objective,
        val_questions=val,
        val_pools=vpools,
        generations=args.generations,
        pop=args.pop,
        progress=_progress,
    )

    # crowd benchmark on the SAME splits (-Brier for calibration; 0 break-even for return).
    def _crowd(xs: list[pe.Question]) -> float:
        pairs = [
            (float(q.market_price), bool(q.outcome))
            for q in xs
            if q.market_price is not None and q.outcome is not None
        ]
        if not pairs:
            return float("nan")
        return -_brier(pairs) if args.objective == "calibration" else 0.0

    crowd_tr, crowd_va = _crowd(train), _crowd(val)
    lift = res.val_fitness - res.seed_val_fitness
    gap = res.val_fitness - crowd_va  # higher is better in both metrics
    verdict = (
        color(f"beats crowd by {gap:+.3f} on holdout", "green")
        if gap > 0
        else color(f"{abs(gap):.3f} behind crowd on holdout", "red")
    )
    power = "" if len(val) >= 40 else color(f"  (n={len(val)} - under rubric power floor)", "dim")

    print(f"\n  climb  {spark(hist_va)}")
    print(
        panel(
            [
                f"{'':10}{'train':>9}  {'holdout':>9}",
                f"crowd     {crowd_tr:>+9.3f}  {crowd_va:>+9.3f}   {color('<- benchmark', 'dim')}",
                f"seed      {res.seed_train_fitness:>+9.3f}  {res.seed_val_fitness:>+9.3f}",
                f"champion  {res.train_fitness:>+9.3f}  {res.val_fitness:>+9.3f}   "
                f"{color(f'{lift:+.3f} vs seed', 'cyan')}",
                f"verdict   {verdict}{power}",
            ],
            title=f"result  -  {metric}",
        )
    )

    changed = [
        (k, v)
        for k, v in vars(res.knobs).items()
        if k not in ("system_prompt", "anthropic_api_key", "model_id")
        and v != getattr(knobs, k, None)
    ]
    if changed:
        print("  evolved  " + "  ".join(f"{k}: {getattr(knobs, k)}->{v}" for k, v in changed))
    else:
        print("  evolved  (seed unchanged - it was already the champion)")


def cmd_traces(cfg: Config, args: argparse.Namespace) -> None:
    if args.market:
        _query(
            cfg,
            """
            SELECT id, created_at, model_name, market_external_id, latency_ms,
                   input_tokens, output_tokens, cache_read_tokens,
                   estimated_cost_usd, error
            FROM llm_calls WHERE market_external_id = %s
            ORDER BY id DESC LIMIT %s
            """,
            (args.market, args.limit),
        )
    else:
        _query(
            cfg,
            """
            SELECT id, created_at, model_name, market_external_id, latency_ms,
                   input_tokens, output_tokens, cache_read_tokens,
                   estimated_cost_usd, error
            FROM llm_calls ORDER BY id DESC LIMIT %s
            """,
            (args.limit,),
        )


def cmd_scout(cfg: Config, args: argparse.Namespace) -> None:
    """Efficiency map: category x thinness. Live, read-only Gamma scan (WP5)."""
    from polyevolve.cli_ui import color, table
    from polyevolve.scout.efficiency import _f, _money, efficiency_map

    cats = args.categories or None
    rows = efficiency_map(cats, limit_per_tag=args.limit_per_tag, top_n=args.top_n)
    headers = ["category", "thin", "n", "med_liq", "med_vol", "spread", "liqT", "volT", "sprT"]
    trows = [
        [
            r.category,
            _f(r.thinness),
            str(r.n_markets),
            _money(r.median_liquidity).strip(),
            _money(r.median_volume).strip(),
            _f(r.mean_spread),
            _f(r.liquidity_thinness, 2),
            _f(r.volume_thinness, 2),
            _f(r.spread_thinness, 2),
        ]
        for r in rows
    ]
    print()
    print(table(headers, trows, title="polymarket scout  -  category x thinness"))
    print(color("  thin = inefficiency proxy in [0,1] (1=thinnest); hunt top rows first.", "dim"))
    leads = [r for r in rows if r.thinnest]
    if leads:
        print(color("\n  thinnest individual markets (concrete leads):", "bold"))
        for r in leads:
            print(color(f"  [{r.category}]", "cyan"))
            for m in r.thinnest:
                print(
                    f"    thin={_f(m.thinness)} liq={_money(m.liquidity).strip()} "
                    f"vol={_money(m.volume).strip()} spr={_f(m.spread)}  {m.question[:54]}"
                )


def cmd_run(cfg: Config, args: argparse.Namespace) -> None:
    """One experiment through the harness: pull -> research -> predict -> score.

    Resolves a recipe of {market, connectors, forecaster} against the live plugin
    registry, runs it through ``harness.run_experiment`` PRICE-FREE and point-in-
    time, prints each forecast + the rubric verdict, and (unless ``--no-ledger``)
    logs every market that diverges from the crowd into the forward paper ledger.
    """
    from polyevolve.harness.rubric import evaluate
    from polyevolve.harness.run import run_experiment

    market_filter = MarketFilter(
        category=args.category,
        tags=tuple(args.tags),
        open_only=not args.include_closed,
        resolves_within_days=args.resolves_within_days,
    )
    results = run_experiment(
        market_source_key=args.market,
        market_filter=market_filter,
        connector_keys=args.connectors,
        forecaster_key=args.forecaster,
        lead_days=args.lead_days,
        limit=args.limit,
        edge_type=args.edge_type,
    )

    from polyevolve.cli_ui import color, panel

    print()
    print(
        panel(
            [
                f"market      {args.market}",
                f"forecaster  {args.forecaster}",
                f"connectors  {', '.join(args.connectors) or '(none)'}",
                f"lead        {args.lead_days} days  -  {len(results.markets)} markets forecast",
            ],
            title="polyevolve run",
        )
    )
    print()
    for m in results.markets:
        crowd = f"{m.crowd_prob:.3f}" if m.crowd_prob is not None else " n/a "
        div = f"{m.divergence:+.3f}" if m.divergence is not None else "  n/a "
        divc = "yellow" if (m.divergence is not None and abs(m.divergence) >= 0.05) else "dim"
        print(
            f"  [{m.confidence:>6}] fair={m.fair_prob:.3f} crowd={crowd} "
            f"div={color(div, divc)}  {m.question[:60]}"
        )
        print(color(f"           connectors: {', '.join(m.connectors_used) or '(none)'}", "dim"))

    report = evaluate(results)
    pass_n = sum(1 for c in report.checks if c.passed)
    rows = []
    for c in report.checks:
        mark = color("PASS", "green") if c.passed else color("FAIL", "red")
        rows.append(f"{mark}  {c.name}: {c.detail}")
    verdict_c = "green" if pass_n == len(report.checks) else "yellow"
    rows.append(color(f"{pass_n}/{len(report.checks)} checks passed", verdict_c))
    print()
    print(panel(rows, title="rubric"))

    if args.no_ledger:
        print("\n(--no-ledger: nothing logged to the forward ledger)")
        return

    logged = _log_to_ledger(cfg, results)
    print(
        f"\nlogged {logged} divergent forecast(s) to the forward paper ledger "
        f"(polyevolve ledger report)."
    )


def _log_to_ledger(cfg: Config, results: Any) -> int:
    """Insert every market with a crowd price into the forward paper ledger.

    Reuses the proven ``paper_bets`` schema/insert so a harness run feeds the one
    un-foolable gate directly. Only markets that carry a crowd price (so a real
    fair-vs-crowd bet exists) are logged; the ledger grades them at resolution.
    """
    from polyevolve.ledger import forward_ledger as fl

    inserts = [m for m in results.markets if m.crowd_prob is not None]
    if not inserts:
        return 0
    with psycopg.connect(cfg.db_url) as conn, conn.cursor() as cur:
        fl.ensure_table(cur)
        for m in inserts:
            res_date = m.end_date.date() if m.end_date is not None else None
            cur.execute(
                """
                INSERT INTO paper_bets (category, market_external_id, question,
                    crowd_price, fair_estimate, rule, confidence, resolution_date, notes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    m.category,
                    m.external_id,
                    m.question,
                    m.crowd_prob,
                    m.fair_prob,
                    results.forecaster_key,
                    m.confidence,
                    res_date,
                    f"harness:{results.market_source_key} lead={results.lead_days}d",
                ),
            )
        conn.commit()
    return len(inserts)


def cmd_ledger(cfg: Config, args: argparse.Namespace) -> None:
    """Forward paper-bet ledger - delegates to the forward_ledger module.

    Forwards the ledger subcommand (init/log/grade/report) and its flags through
    unchanged so the proven ledger CLI stays the single source of truth.
    """
    from polyevolve.ledger import forward_ledger

    sys.argv = ["forward_ledger", *args.ledger_args]
    raise SystemExit(forward_ledger.main())


def main() -> int:
    parser = argparse.ArgumentParser(prog="polyevolve")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scout", help="efficiency map: live category x thinness scan")
    p.add_argument(
        "categories",
        nargs="*",
        help="category tag_slugs to scan (default: politics geopolitics world ...)",
    )
    p.add_argument("--limit-per-tag", dest="limit_per_tag", type=int, default=100)
    p.add_argument("--top-n", dest="top_n", type=int, default=5)
    p.set_defaults(func=cmd_scout)

    p = sub.add_parser("run", help="run one experiment through the harness into the ledger")
    p.add_argument("--market", default="polymarket", help="market source key (default: polymarket)")
    p.add_argument(
        "--forecaster", default="baseline", help="forecaster key (default: baseline, no GPU)"
    )
    p.add_argument("--connectors", nargs="*", default=[], help="research connector keys, in order")
    p.add_argument("--category", default=None, help="market category filter")
    p.add_argument("--tags", nargs="*", default=[], help="match markets carrying ANY of these tags")
    p.add_argument("--lead-days", dest="lead_days", type=int, default=30, help="forecast horizon")
    p.add_argument("--limit", type=int, default=1, help="max markets to process (default: 1)")
    p.add_argument("--resolves-within-days", dest="resolves_within_days", type=int, default=None)
    p.add_argument("--include-closed", action="store_true", help="include closed markets")
    p.add_argument(
        "--edge-type", dest="edge_type", default=None, help="claimed edge type (rubric check 5)"
    )
    p.add_argument(
        "--no-ledger", action="store_true", help="do not log forecasts to the forward ledger"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ledger", help="forward paper-bet ledger (delegates to forward_ledger)")
    p.add_argument(
        "ledger_args",
        nargs=argparse.REMAINDER,
        help="subcommand + flags passed through to forward_ledger (init/log/grade/report)",
    )
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("predictions", help="recent predictions with market context")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_predictions)

    p = sub.add_parser("calibration", help="decile calibration + edge over market")
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser("cost", help="token usage + estimated cost")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("runs", help="per-day run summary")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("coverage", help="market coverage by status")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("traces", help="recent LLM calls")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--market", type=str, default=None, help="filter by market external id")
    p.set_defaults(func=cmd_traces)

    p = sub.add_parser("backtest", help="backtest calibration (clean vs contaminated)")
    p.add_argument("--run", type=str, default=None, help="run_id (default: latest)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("snapshot", help="build a resolved-market dataset to evolve against")
    p.add_argument("--set", dest="snapshot_set", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--lead-days", type=int, default=7)
    p.add_argument("--max-per-event", type=int, default=3)
    p.add_argument(
        "--domain",
        default="foreign_politics",
        help="ingestion domain: foreign_politics | sports | crypto | culture | all",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="discovery liquidity gate: skip markets with volume below this ($)",
    )
    p.add_argument(
        "--no-research",
        action="store_true",
        help="skip per-market research gathering (fast price+outcome dataset)",
    )
    p.add_argument(
        "--pages-per-tag", type=int, default=10, help="event pages to pull per tag (raise for n)"
    )
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("snapshots", help="list frozen eval snapshot sets")
    p.set_defaults(func=cmd_snapshots)

    p = sub.add_parser("evolve", help="evolve a strategy genome; seed -> champion on a holdout")
    p.add_argument("--source", default="polymarket", choices=["polymarket", "manifold"])
    p.add_argument(
        "--snapshot-set",
        dest="snapshot_set",
        default=None,
        help="resolved-market set to evolve on (for --source polymarket)",
    )
    p.add_argument("--path", default=None, help="manifold jsonl path (for --source manifold)")
    p.add_argument(
        "--objective",
        default="calibration",
        choices=["calibration", "return"],
        help="calibration = -Brier; return = net-of-spread event ROI",
    )
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--pop", type=int, default=6)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--holdout", type=float, default=0.25, help="validation fraction (held out)")
    p.add_argument("--gather", action="store_true", help="gather leakage-safe evidence pools")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("evaluate", help="evaluate default genome on a snapshot set")
    p.add_argument("--set", dest="snapshot_set", required=True)
    p.add_argument("--model", type=str, default="ollama/qwen2.5:14b")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sweep", help="evaluate one genome across multiple models")
    p.add_argument("--set", dest="snapshot_set", required=True)
    p.add_argument("--models", nargs="+", required=True, help="model ids to sweep")
    p.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    cfg = Config.from_env()
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
