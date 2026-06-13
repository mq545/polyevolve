"""Evaluate a (genome, model) candidate against a frozen eval snapshot.

The fitness function for fast iteration AND evolution. Network-free: it reads
frozen research context + historical price + outcome from eval_snapshots, and
caches each prediction in prediction_cache keyed on (snapshot, model, genome).
Re-evaluating an unchanged candidate is a pure cache read - no LLM calls.

Contamination + holdout are computed PER MODEL here (the snapshot is model-
agnostic): a market is clean iff it resolved after the model's cutoff + margin;
clean markets are split train/holdout deterministically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg

from polyevolve.agents.foreign_politics_agent import PREDICTION_TOOL
from polyevolve.contracts import Model
from polyevolve.evolution.genome import Genome
from polyevolve.models.cutoffs import is_clean_for_backtest
from polyevolve.orchestration.scoring import assign_splits, brier

logger = logging.getLogger(__name__)


# Complexity penalty on combined_score: ~1e-6 / genome char. A lean ~2-3k-char
# genome costs ~0.002-0.003 Brier; a baroque 10k-char "graph wave forecaster" costs
# ~0.01 - enough to lose a tie, not enough to swamp a real edge. Stops the prompt
# ballooning that let past genomes overfit via verbosity. Tune if needed.
_COMPLEXITY_LAMBDA = 1e-6


@dataclass
class EvalResult:
    model_name: str
    genome_hash: str
    n_total: int
    n_clean: int
    # clean-cohort metrics, split out (holdout is the trustworthy one)
    brier_train: float | None
    brier_holdout: float | None
    # edge vs market on priced clean markets
    edge_train: float | None
    edge_holdout: float | None
    n_priced_clean: int
    cache_hits: int
    cache_misses: int
    # Pristine TEST set: NEVER feeds combined_score / selection. Scored for
    # reporting only (None when test_frac=0, i.e. two-way mode). The champion's
    # edge_test is the one trustworthy "did we beat the market" number.
    brier_test: float | None = None
    edge_test: float | None = None
    n_test: int = 0
    failed: int = 0
    # brier_cv = mean Brier over ALL non-test (train+holdout) markets - the denoised
    # fitness signal. genome_chars feeds the complexity penalty.
    brier_cv: float | None = None
    genome_chars: int = 0
    per_market: list[dict[str, Any]] = field(default_factory=list)

    @property
    def combined_score(self) -> float:
        """Fitness ShinkaEvolve MAXIMIZES = -(CV Brier) - complexity_penalty.

        The forecaster fits NOTHING in-eval, so train/holdout Briers are
        statistically exchangeable - validating on a 30% holdout (n~44) threw away
        70% of the signal and let selection chase noise (SE 0.031). We instead
        validate on ALL non-test markets (brier_cv, SE ~0.21/sqrt(n)) - the
        anti-overfit denoising. The pristine TEST set never enters here; the
        cv-vs-test gap IS the overfit monitor. The complexity term discourages the
        baroque prompt growth that was a past overfit channel. (True k-fold only
        becomes necessary if we add an in-eval fitted layer e.g. calibration.)
        """
        b = self.brier_cv
        if b is None:
            b = self.brier_holdout if self.brier_holdout is not None else self.brier_train
        if b is None:
            return -1.0
        return -b - _COMPLEXITY_LAMBDA * self.genome_chars


def _build_user_content(genome: Genome, question: str, context: dict[str, str]) -> str:
    sections = [f"MARKET QUESTION: {question}"]
    for source, rendered in context.items():
        weight = genome.data_weights.get(source, 0.0)
        if weight <= 0:
            continue
        cap = max(500, int(genome.max_context_chars * min(weight, 1.0)))
        sections.append(f"\n--- [{source}] ---\n{rendered[:cap]}")
    # Fail loud on phantom sources: the mutator has invented data_weights keys
    # that don't exist (e.g. "local_language_analysis"), which previously
    # silently no-op'd - a genome could "use" a source that contributes nothing
    # and never be penalized for it. Surface any weighted-but-absent source so
    # the model sees the gap and the genome can't get a free pass.
    missing = [s for s, w in genome.data_weights.items() if w > 0 and s not in context]
    for s in missing:
        sections.append(f"\n--- [{s}] ---\n[SOURCE ERROR] requested source '{s}' is not available")
    sections.append(
        "\nProduce a calibrated probability. Call submit_prediction with your estimate."
    )
    return "\n".join(sections)


def _load_snapshot(conn: psycopg.Connection, snapshot_set: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, market_external_id, question, as_of, resolved_at, outcome,
                   market_price_at_as_of, research_context
            FROM eval_snapshots WHERE snapshot_set = %s ORDER BY id
            """,
            (snapshot_set,),
        )
        cols = [d.name for d in cur.description or []]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def _cached_prediction(
    conn: psycopg.Connection, snapshot_id: int, model_name: str, genome_hash: str
) -> float | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT probability_yes FROM prediction_cache
            WHERE snapshot_id = %s AND model_name = %s AND genome_hash = %s
            """,
            (snapshot_id, model_name, genome_hash),
        )
        row = cur.fetchone()
        return float(row[0]) if row else None


def _store_prediction(
    conn: psycopg.Connection,
    snapshot_id: int,
    model_name: str,
    genome_hash: str,
    prob: float,
    confidence: float | None,
    reasoning: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_cache
                (snapshot_id, model_name, genome_hash, probability_yes, confidence, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (snapshot_id, model_name, genome_hash) DO NOTHING
            """,
            (snapshot_id, model_name, genome_hash, prob, confidence, reasoning),
        )


def evaluate(
    *,
    genome: Genome,
    model: Model,
    db_url: str,
    snapshot_set: str,
    holdout_frac: float = 0.3,
    test_frac: float = 0.0,
) -> EvalResult:
    ghash = genome.hash()
    cached_blocks = [genome.system_prompt, genome.domain_context]

    from polyevolve.storage import db

    with db.connection(db_url) as conn:
        rows = _load_snapshot(conn, snapshot_set)
        if not rows:
            raise RuntimeError(f"snapshot set {snapshot_set!r} is empty - build it first")

        clean_flags = {
            r["market_external_id"]: is_clean_for_backtest(model.name, r["resolved_at"])
            for r in rows
        }
        clean_ids = [mid for mid, c in clean_flags.items() if c]
        splits = assign_splits(clean_ids, holdout_frac=holdout_frac, test_frac=test_frac)

        hits = misses = failed = 0
        per_market: list[dict[str, Any]] = []

        for r in rows:
            mid = r["market_external_id"]
            prob = _cached_prediction(conn, r["id"], model.name, ghash)
            confidence: float | None = None
            reasoning: str | None = None
            if prob is None:
                misses += 1
                context = r["research_context"] or {}
                user_content = _build_user_content(genome, r["question"], context)
                # Fail-soft: a single market that won't parse / errors must not
                # crash a 150-market run. Skip it and count it.
                try:
                    result = model.complete_with_tool(
                        cached_system_blocks=cached_blocks,
                        user_content=user_content,
                        tool=PREDICTION_TOOL,
                        metadata={"genome_hash": ghash, "market_external_id": mid},
                    )
                    pred = result["input"]
                    prob = float(pred["probability_yes"])
                except Exception:
                    failed += 1
                    logger.warning("prediction failed for market %s - skipping", mid)
                    continue
                confidence = float(pred.get("confidence", 0.0))
                reasoning = str(pred.get("reasoning", ""))[:4000]
                _store_prediction(conn, r["id"], model.name, ghash, prob, confidence, reasoning)
                conn.commit()
            else:
                hits += 1

            price = r["market_price_at_as_of"]
            per_market.append(
                {
                    "market_id": mid,
                    "outcome": r["outcome"],
                    "prob": prob,
                    "price": float(price) if price is not None else None,
                    "is_clean": clean_flags[mid],
                    "split": splits.get(mid),
                    "brier_agent": brier(prob, r["outcome"]),
                    "brier_market": (
                        brier(float(price), r["outcome"]) if price is not None else None
                    ),
                }
            )

    genome_chars = len(genome.system_prompt) + len(genome.domain_context)
    return _aggregate(
        model.name, ghash, rows, per_market, hits, misses, failed, genome_chars=genome_chars
    )


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _aggregate(
    model_name: str,
    ghash: str,
    rows: list[dict[str, Any]],
    per_market: list[dict[str, Any]],
    hits: int,
    misses: int,
    failed: int = 0,
    genome_chars: int = 0,
) -> EvalResult:
    clean = [m for m in per_market if m["is_clean"]]
    train = [m for m in clean if m["split"] == "train"]
    holdout = [m for m in clean if m["split"] == "holdout"]
    test = [m for m in clean if m["split"] == "test"]
    nontest = train + holdout  # the full validation set fitness is scored on

    def edge(ms: list[dict[str, Any]]) -> float | None:
        priced = [m for m in ms if m["brier_market"] is not None]
        if not priced:
            return None
        agent = _mean([m["brier_agent"] for m in priced])
        market = _mean([m["brier_market"] for m in priced])
        return None if agent is None or market is None else market - agent

    n_priced_clean = sum(1 for m in clean if m["brier_market"] is not None)

    return EvalResult(
        model_name=model_name,
        genome_hash=ghash,
        n_total=len(per_market),
        n_clean=len(clean),
        brier_train=_mean([m["brier_agent"] for m in train]),
        brier_holdout=_mean([m["brier_agent"] for m in holdout]),
        edge_train=edge(train),
        edge_holdout=edge(holdout),
        n_priced_clean=n_priced_clean,
        cache_hits=hits,
        cache_misses=misses,
        brier_test=_mean([m["brier_agent"] for m in test]),
        edge_test=edge(test),
        n_test=len(test),
        failed=failed,
        brier_cv=_mean([m["brier_agent"] for m in nontest]),
        genome_chars=genome_chars,
        per_market=per_market,
    )
