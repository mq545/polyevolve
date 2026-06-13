"""The 8 ARCHITECTURE.md rubric checks, as PURE functions.

Every experiment is scored on these - the rubric is the EXPLORATION-zone gate that
discards an idea before it can earn (false) belief. Each check is a pure function
taking the relevant facts and returning ``(passed: bool, detail: str)``: no I/O, no
hidden state, trivially unit-testable. ``evaluate(results)`` runs the subset that
can be derived from a harness run and assembles a :class:`RubricReport`.

The checks are earned from real failures (see ARCHITECTURE.md):
  1. Power                  - n>=~40 AND edge >= 2 SE          (our #1 killer)
  2. Out-of-sample/forward  - confirmed on fresh/future data   (overfit race edge)
  3. Executable             - survives an order-book WALK at size after spread
  4. Multiple-testing       - forward-confirmed + FDR aware, not in-sample p
  5. Edge-type named        - predictive/structural/latency/calibration/artifact
  6. Data real & PIT        - machine-readable, no leakage     (chart-image polls)
  7. Observation-reviewed   - reasoning traces actually read   (caught hallucination)
  8. Inefficiency x edge    - category fit                     (politics efficient)

A check returning ``passed=False`` is not a crash - it is the rubric doing its job.
``evaluate`` never raises on a bad experiment; it reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The named edge types from ARCHITECTURE.md check 5. An experiment must claim one.
EDGE_TYPES: tuple[str, ...] = (
    "predictive",
    "structural",
    "latency",
    "calibration",
    "resolution-artifact",
)

# Minimum resolved sample size before any GO/NO-GO is allowed (check 1).
MIN_POWER_N = 40
# Minimum signal-to-noise (edge in standard errors) before an edge is real.
MIN_POWER_SE = 2.0


# --------------------------------------------------------------------------- #
# The 8 checks - each a pure function returning (passed, detail).
# --------------------------------------------------------------------------- #
def check_power(n: int, edge: float, se: float) -> tuple[bool, str]:
    """1. Power - refuse a verdict on n<~40 or an edge under ~2 SE.

    Sample size has been our #1 killer: a handful of resolved markets can show a
    large edge that is pure noise. ``edge``/``se`` are in the same units (e.g.
    Brier improvement). A non-positive or unknown ``se`` cannot support an edge.
    """
    if n < MIN_POWER_N:
        return False, f"underpowered: n={n} < {MIN_POWER_N}"
    if se <= 0.0 or not math.isfinite(se):
        return False, f"no usable standard error (se={se})"
    ratio = abs(edge) / se
    if ratio < MIN_POWER_SE:
        return False, f"edge {edge:+.4f} is only {ratio:.2f} SE (< {MIN_POWER_SE})"
    return True, f"n={n}, edge {edge:+.4f} = {ratio:.2f} SE"


def check_out_of_sample(forward_or_oos: bool) -> tuple[bool, str]:
    """2. Out-of-sample / forward - never confirm on the data the edge was found on.

    A harness run is in-sample by construction (EXPLORATION zone), so this is a
    declared flag the experiment carries: it is only True once the edge is being
    confirmed on fresh/disjoint/future data (the forward ledger).
    """
    if forward_or_oos:
        return True, "confirmed on fresh/forward/out-of-sample data"
    return False, "in-sample only - EXPLORATION result, belief = 0"


def check_executable(
    order_book: tuple[tuple[float, float], ...],
    side: str,
    fair_prob: float,
    size: float,
) -> tuple[bool, str]:
    """3. Executable - does the edge survive an order-book WALK at real ``size``?

    Walks ``order_book`` (best level first) filling ``size`` shares and computes
    the size-weighted average fill price - the price you actually pay, after the
    spread, not a mid-quote artifact. The edge is executable only if the average
    fill is still on the right side of our ``fair_prob``:
      - buying YES (``side='YES'``): avg fill price < fair_prob (room to profit)
      - buying NO  (``side='NO'``):  avg fill (1-price) < (1-fair_prob)
    Insufficient depth to fill ``size`` => not executable (None-book => empty).

    ``order_book`` levels are ``(price, size)`` in the OrderBook convention: asks
    best-first for a YES buy, bids best-first for a NO buy (caller passes the side
    being lifted). Prices are YES-share prices in [0, 1].
    """
    if size <= 0.0:
        return False, "non-positive size requested"
    if not order_book:
        return False, "empty book - not executable"

    remaining = size
    cost = 0.0
    for price, level_size in order_book:
        take = min(remaining, level_size)
        cost += take * price
        remaining -= take
        if remaining <= 0.0:
            break
    if remaining > 0.0:
        filled = size - remaining
        return False, f"insufficient depth: only {filled:.1f}/{size:.1f} shares available"

    avg_fill = cost / size
    if side == "YES":
        ok = avg_fill < fair_prob
        margin = fair_prob - avg_fill
        return ok, f"YES avg fill {avg_fill:.4f} vs fair {fair_prob:.4f} (margin {margin:+.4f})"
    if side == "NO":
        # Cost is in YES-share-price terms; the NO buyer pays (1 - yes_price).
        no_fill = 1.0 - avg_fill
        no_fair = 1.0 - fair_prob
        ok = no_fill < no_fair
        margin = no_fair - no_fill
        return ok, f"NO avg fill {no_fill:.4f} vs fair {no_fair:.4f} (margin {margin:+.4f})"
    return False, f"unknown side {side!r} (expected 'YES' or 'NO')"


def check_multiple_testing(forward_confirmed: bool, fdr_aware: bool) -> tuple[bool, str]:
    """4. Multiple-testing discipline - promotion needs forward confirmation + FDR
    awareness, never an in-sample p-value. Running many experiments manufactures
    false positives; the defense is structural, not a threshold.
    """
    if forward_confirmed and fdr_aware:
        return True, "forward-confirmed with FDR awareness"
    missing = []
    if not forward_confirmed:
        missing.append("forward confirmation")
    if not fdr_aware:
        missing.append("FDR awareness")
    return False, "missing " + " + ".join(missing)


def check_edge_type_named(edge_type: str | None) -> tuple[bool, str]:
    """5. Edge-type named - the experiment must claim ONE of the known kinds, so we
    know what we think we are exploiting (and can sanity-check it).
    """
    if edge_type in EDGE_TYPES:
        return True, f"edge type = {edge_type}"
    return False, f"edge type {edge_type!r} not in {EDGE_TYPES}"


def check_data_real_pit(machine_readable: bool, no_leakage: bool) -> tuple[bool, str]:
    """6. Data is real & point-in-time - machine-readable (not a chart image) and
    leakage-guarded (only data strictly before ``as_of``). No method fixes
    unextractable or future-leaked inputs.
    """
    if machine_readable and no_leakage:
        return True, "machine-readable and point-in-time (no leakage)"
    problems = []
    if not machine_readable:
        problems.append("not machine-readable")
    if not no_leakage:
        problems.append("leakage suspected")
    return False, "; ".join(problems)


def check_observation_reviewed(reviewed: bool, n_traces: int) -> tuple[bool, str]:
    """7. Observation-level reviewed - someone read the actual reasoning traces, not
    just the aggregate Brier. This is how we caught hallucination-on-garbage.
    """
    if not reviewed:
        return False, "reasoning traces not yet read"
    if n_traces <= 0:
        return False, "marked reviewed but no traces present to review"
    return True, f"{n_traces} reasoning trace(s) reviewed"


def check_inefficiency_advantage(inefficiency: float, advantage: float) -> tuple[bool, str]:
    """8. Inefficiency x our-advantage - category fit. Both must be positive: an
    efficient category (politics) or one we have no edge in is a poor bet even if
    the in-sample number looks good. Inputs are normalized scores in [0, 1].
    """
    product = inefficiency * advantage
    fit = f"inefficiency {inefficiency:.2f} x advantage {advantage:.2f}"
    if inefficiency > 0.0 and advantage > 0.0:
        return True, f"fit = {fit} = {product:.2f}"
    return False, f"poor fit: {fit} (need both > 0)"


# --------------------------------------------------------------------------- #
# Report assembly.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckResult:
    """One rubric check's outcome: its name, pass/fail, and a human-readable why."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RubricReport:
    """The full rubric verdict for an experiment.

    ``passed`` is the AND of every *applicable* check. Checks that cannot be
    derived from a single in-sample harness run (forward/OOS, multiple-testing,
    observation review) are reported as failing-by-default with an explanatory
    detail - an EXPLORATION result legitimately does not clear the CONFIRMATION
    gate, and the report says so rather than silently passing.
    """

    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def summary(self) -> str:
        n_pass = sum(c.passed for c in self.checks)
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict} ({n_pass}/{len(self.checks)} checks)"


def evaluate(results: ExperimentResults) -> RubricReport:
    """Score a harness :class:`ExperimentResults` against all 8 rubric checks.

    Pure: derives every check from ``results`` plus the experiment's declared
    metadata, never touches the network or DB. The forward/OOS, multiple-testing,
    and observation-review checks read declared flags off the results (default
    False) - a raw EXPLORATION run will fail them, which is correct: belief is
    earned only later in the forward ledger.
    """
    n = results.n_resolved
    checks: list[CheckResult] = []

    passed, detail = check_power(n, results.edge, results.edge_se)
    checks.append(CheckResult("power", passed, detail))

    passed, detail = check_out_of_sample(results.forward_or_oos)
    checks.append(CheckResult("out_of_sample", passed, detail))

    passed, detail = check_executable(
        results.executable_book,
        results.executable_side,
        results.executable_fair_prob,
        results.executable_size,
    )
    checks.append(CheckResult("executable", passed, detail))

    passed, detail = check_multiple_testing(results.forward_confirmed, results.fdr_aware)
    checks.append(CheckResult("multiple_testing", passed, detail))

    passed, detail = check_edge_type_named(results.edge_type)
    checks.append(CheckResult("edge_type_named", passed, detail))

    passed, detail = check_data_real_pit(results.data_machine_readable, results.data_no_leakage)
    checks.append(CheckResult("data_real_pit", passed, detail))

    passed, detail = check_observation_reviewed(results.observation_reviewed, results.n_traces)
    checks.append(CheckResult("observation_reviewed", passed, detail))

    passed, detail = check_inefficiency_advantage(results.inefficiency, results.advantage)
    checks.append(CheckResult("inefficiency_advantage", passed, detail))

    return RubricReport(checks=tuple(checks))


# Late import for the type used in evaluate's signature; defined in run.py. Kept
# at the bottom (and guarded for type-checkers) so rubric.py stays importable on
# its own - the checks above have no dependency on the harness runner.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from polyevolve.harness.run import ExperimentResults
