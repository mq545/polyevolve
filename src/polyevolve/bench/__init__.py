"""The BENCH - the fitness oracle that scores genomes.

A genome (`polyevolve.reason.dsl.Genome`) is a `(Question, EvidencePool) -> Forecast`
callable. The bench runs it over a corpus of RESOLVED, point-in-time questions
and scores its probabilities against ground truth.

Public surface:
  - load_manifold / load_manifold_with_pools : dev-corpus loaders (datasets.py)
  - brier / ece / calibration_curve          : calibration scorers (scoring.py)
  - temporal_split / event_cluster           : leakage-safe splits (splits.py)
  - evaluate_calibration                     : run a genome -> {'brier','ece','n'}

The genome NEVER sees `Question.outcome` / `crowd_prob` / `market_price`; only
the bench reads them when scoring.

A second fitness axis - net-of-spread realized PnL, not just calibration - lives in
`returns.py` as `evaluate_return(genome, questions, pools, ...)`, which runs forecasts
through the adversarial market simulator in `sim.py` (`run_adversarial_sim`, `Trade`,
the cost models) rather than re-implementing fills/spread. Calibration (here) and return
(there) are deliberately separate fitness signals.
"""

from __future__ import annotations

from collections.abc import Sequence

from polyevolve.reason.dsl import EvidencePool, Genome, Question

from .datasets import load_manifold, load_manifold_with_pools, parse_question, pool_for
from .scoring import Pair, brier, calibration_curve, ece
from .splits import Split, event_cluster, temporal_split

__all__ = [
    "load_manifold",
    "load_manifold_with_pools",
    "parse_question",
    "pool_for",
    "brier",
    "ece",
    "calibration_curve",
    "Pair",
    "temporal_split",
    "event_cluster",
    "Split",
    "evaluate_calibration",
    "evaluate_calibration_selective",
]


def evaluate_calibration(
    genome: Genome,
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
) -> dict[str, float]:
    """Run `genome` over each (question, pool) and score calibration vs outcome.

    Returns ``{'brier', 'ece', 'n'}`` where ``n`` is the number of questions that
    had a usable ground-truth outcome. Questions with ``outcome is None`` are
    skipped (they cannot be scored). If ``pools`` is omitted, an empty pool is
    passed for every question (genomes that only need the question text still run).

    The genome is treated as untrusted: any exception it raises is swallowed and
    that question is skipped, so one bad program cannot crash a whole sweep.
    """
    if pools is not None and len(pools) != len(questions):
        raise ValueError("pools must align 1:1 with questions when provided")

    pairs: list[Pair] = []
    for i, q in enumerate(questions):
        if q.outcome is None:
            continue
        pool = pools[i] if pools is not None else EvidencePool(items=[])
        try:
            forecast = genome(q, pool)
        except Exception:
            continue
        pairs.append((float(forecast.p_yes), bool(q.outcome)))

    return {
        "brier": brier(pairs),
        "ece": ece(pairs),
        "n": float(len(pairs)),
    }


def evaluate_calibration_selective(
    genome: Genome,
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
    *,
    min_confidence: float = 0.0,
    require_trade: bool = False,
) -> dict[str, float]:
    """Calibration with a REJECT OPTION: score Brier only on the markets the genome COVERS.

    A market is *covered* if the genome's ``confidence >= min_confidence`` (and, when
    ``require_trade``, it also chose to trade: ``size != 0``). Returns both the covered
    Brier/ECE and the all-markets Brier, plus ``coverage`` - so we can read the
    precision/coverage tradeoff (does declining the low-confidence markets sharpen the rest?).

    The threshold MUST be chosen on a train split and applied blind to holdout; picking it
    post-hoc on the same data manufactures fake selectivity gains. ``coverage`` is always
    reported so a low-coverage policy is never mistaken for a broadly-accurate one.
    """
    if pools is not None and len(pools) != len(questions):
        raise ValueError("pools must align 1:1 with questions when provided")

    all_pairs: list[Pair] = []
    covered: list[Pair] = []
    for i, q in enumerate(questions):
        if q.outcome is None:
            continue
        pool = pools[i] if pools is not None else EvidencePool(items=[])
        try:
            fc = genome(q, pool)
        except Exception:
            continue
        pair = (float(fc.p_yes), bool(q.outcome))
        all_pairs.append(pair)
        is_covered = float(fc.confidence) >= min_confidence and (
            not require_trade or float(fc.size) != 0.0
        )
        if is_covered:
            covered.append(pair)

    n_total = len(all_pairs)
    return {
        "brier": brier(covered),
        "ece": ece(covered),
        "brier_all": brier(all_pairs),
        "n_covered": float(len(covered)),
        "n_total": float(n_total),
        "coverage": (len(covered) / n_total) if n_total else 0.0,
    }
