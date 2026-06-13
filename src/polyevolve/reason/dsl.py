"""The genome contract - the keystone every other module compiles against.

A **genome** is an evolvable function `forecast(Question, EvidencePool) -> Forecast`,
composed from a vocabulary of typed **Node**s (see polyevolve.reason.nodes). ShinkaEvolve
mutates the *composition* (the body of the seed program); the nodes are the stable,
unit-tested primitives.

Design rules (do not break - agents build against this):
- Everything flows through `ReasoningState` (immutable-ish: nodes return a new/updated copy).
- A Node is any callable `(ReasoningState) -> ReasoningState`. Nodes are small, typed,
  individually testable, and never reach the network except via injected clients.
- The final `Forecast` is read off `state.beliefs` by the genome's return.
- Leakage rule: nodes may only use evidence with `date <= question.as_of`. The pool is
  pre-filtered, but date-respecting nodes must re-check.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """One piece of point-in-time evidence."""

    text: str
    source: str = ""
    date: datetime | None = None  # publication date; must be <= question.as_of to be usable


class EvidencePool(BaseModel):
    """The FROZEN, over-gathered, leakage-audited evidence corpus for one question.

    Acquisition is cached upstream; the genome's retrieval/selection nodes evolve over
    THIS pool (features-as-fetched frozen; features-as-used evolved).
    """

    items: list[EvidenceItem] = Field(default_factory=list)

    def on_or_before(self, as_of: datetime) -> list[EvidenceItem]:
        """Leakage guard: only evidence dated on/before the as-of cutoff (undated kept,
        flagged by the caller's discipline)."""
        return [e for e in self.items if e.date is None or e.date <= as_of]


class Question(BaseModel):
    """A binary forecasting question with a point-in-time cutoff."""

    id: str
    text: str
    as_of: datetime  # T - forecast as if "today" is this date; evidence must be <= T
    resolution_criteria: str = ""  # exact YES condition (parsing this well matters - see nodes)
    category: str = ""
    # ground truth + market context (NEVER visible to the genome; used only by bench/):
    outcome: bool | None = None
    market_price: float | None = None
    crowd_prob: float | None = None
    # execution metadata for the net-of-spread RETURN scorer (bench/returns.py); all
    # optional so the Manifold/calibration path is unaffected:
    liquidity: float | None = None  # USD; drives the spread cost tier + depth cap
    event_id: str | None = None  # correlated markets share this -> one independent obs
    lead_days: int | None = None  # as_of -> resolution, for regime bucketing

    def blinded(self) -> Question:
        """A copy safe to hand an UNTRUSTED genome: future/answer fields stripped.

        Drops the resolved ``outcome`` (future information) and the ``crowd_prob`` (the
        crowd's own answer) so no genome - including LLM-authored, full-program ones that
        receive the raw object - can read the result and reward-hack a backtest. The bench
        always scores against the TRUE outcome it keeps separately. ``market_price`` is
        retained: it is known at decision time and is needed for net-of-spread sizing
        (the forecaster prompt never renders it - see reason.nodes._question_block).
        """
        return self.model_copy(update={"outcome": None, "crowd_prob": None})


class Forecast(BaseModel):
    """A genome's output for one question."""

    p_yes: float = Field(ge=0.0, le=1.0)
    size: float = 0.0  # signed stake fraction in [-1, 1]; 0.0 == ABSTAIN (no trade)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class ReasoningState(BaseModel):
    """The single object that flows through the node pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    question: Question
    pool: EvidencePool
    beliefs: dict[str, Any] = Field(default_factory=dict)  # e.g. {"p_yes":, "margin_mu":, "sigma":}
    selected: list[EvidenceItem] = Field(default_factory=list)  # evidence the retrieval node kept
    trace: list[str] = Field(default_factory=list)  # auditable record, one line per node

    def log(self, msg: str) -> ReasoningState:
        self.trace.append(msg)
        return self

    def to_forecast(self) -> Forecast:
        """Read the final Forecast off beliefs (defaults are deliberately humble)."""
        return Forecast(
            p_yes=float(self.beliefs.get("p_yes", 0.5)),
            size=float(self.beliefs.get("size", 0.0)),
            confidence=float(self.beliefs.get("confidence", 0.5)),
            rationale=str(self.beliefs.get("rationale", "")),
        )


@runtime_checkable
class Node(Protocol):
    """Any callable that advances the reasoning state. The unit of the genome."""

    def __call__(self, state: ReasoningState) -> ReasoningState: ...


# A genome is an evolvable function from (question, evidence) to a forecast.
# The SEED genome (polyevolve.reason.seed) is one such function with an EVOLVE-BLOCK body
# that ShinkaEvolve mutates. bench/ scores it; evolve/ mutates+selects it.
Genome = Callable[[Question, EvidencePool], Forecast]
