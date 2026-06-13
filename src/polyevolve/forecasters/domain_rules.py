"""Domain-rules forecaster - the Tier-1 electoral through-lines as code.

A deterministic, $0, GPU-free forecaster encoding the political-domain through-lines
the CROWD systematically misprices (project memory: project_polymarket_domain_playbook).
The META-ERROR the crowd makes: it prices the SALIENT, NATIONAL, GENERAL-POPULATION
signal when the contest is decided by a DIFFERENT smaller/skewed/rule-distorted
mechanism knowable in advance from free data. Each rule below detects one form of
that error and nudges a base probability.

Design contract:
  - Each rule is a pure function `f_*(prob, ctx) -> RuleResult` that takes a base
    probability and a parsed-feature dict and returns an (adjusted prob, fired?,
    note) triple. Rules NEVER look at the market price (price-free, like the LLM
    path) and never read the future (point-in-time).
  - `predict` runs the rules in priority order, composing their adjustments, and
    reports the firing rules in `reasoning`. With no parsed features (the common
    case until WP2 connectors emit them), NO rule fires and we return the base
    rate at low confidence - a safe null that degrades to baseline.

Expected feature keys (produced by connectors / a future parse step; all optional):
  resolver_type : "general" | "selectorate" | "runoff" | "by_election"
  poll_house_lean: signed pp the GOVT-aligned pollsters favor the incumbent by,
                   vs the independent mean (independent_mean - house_mean for the
                   challenger; >5 => captured polling tilts to challenger)
  is_captured_regime: bool - Hungary/Serbia/Turkey/Georgia-class hybrid regime
  is_fptp        : bool - first-past-the-post / mixed seat system
  vote_lead_pp   : signed pp lead of the side this market's YES backs
  vote_moe_pp    : polling margin of error in pp (default ~3)
  subject_is_frontrunner: bool - does YES back the public-poll frontrunner?
  prior_yes      : float in [0,1] - reference-class P(YES) the rules fade/tilt
                   (a model-side prior, e.g. seat-leader from poll projection; NOT
                   the market price). Defaults to 0.5 when absent.

IMPORTANT TEMPERING (live forward test, project_polymarket_domain_playbook):
  - Rule C (vote/seat divergence) is the ONE rule that survived a live test, and
    only as a MAGNITUDE FADE toward ~0.6 - NOT a contrarian flip.
  - Rules A (wrong-electorate) and E (incumbent-decay) are TRAPS without a
    consolidation-DIRECTION check (consolidation often runs WITH the leader). They
    are implemented CONSERVATIVELY and gated behind an explicit, defaulted-off
    `consolidation_against_leader` feature, with TODOs, so they never fire on a
    naive reflex.

Self-registers as @register_forecaster("domain_rules"). Core never imports this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polyevolve.core.registry import register_forecaster
from polyevolve.core.types import Confidence, Market, Prediction

# A within-MoE FPTP race priced as a lopsided winner is faded toward this anchor
# (live test: Makerfield 0.735->0.63, Peru 0.76->0.62 - a magnitude correction).
RULE_C_FADE_ANCHOR = 0.6
# Captured-polling challenger gap (pp) above which Rule B tilts to the challenger.
CAPTURED_POLL_GAP_PP = 5.0
# Max probability nudge a single captured-poll firing applies, toward the challenger.
CAPTURED_POLL_MAX_SHIFT = 0.10
DEFAULT_MOE_PP = 3.0


@dataclass(frozen=True)
class RuleResult:
    """One rule's verdict: did it fire, the (possibly) adjusted prob, and why."""

    prob: float
    fired: bool
    note: str


def _clamp(p: float) -> float:
    return min(1.0, max(0.0, p))


def f_captured_poll(prob: float, ctx: dict[str, Any]) -> RuleResult:
    """Rule B (Tier-1) - Captured-polling reweight. HIGHEST PRECISION (3/3 Hungary).

    In hybrid/captured regimes, govt-aligned pollsters overstate the incumbent. If
    the independent mean beats the govt-house mean for the CHALLENGER by more than
    ~5pp, tilt probability toward the challenger. The pollster conflict-of-interest
    table is already in the fetched Wikipedia data; the crowd just doesn't parse it.

    Convention: `poll_house_lean` is (independent_mean - house_mean) for the
    challenger in pp - positive means independents see the challenger stronger than
    captured houses do. We nudge `prob` toward the challenger proportionally. Whether
    YES *is* the challenger is carried by `subject_is_frontrunner` (frontrunner ==
    incumbent here): if YES backs the frontrunner we nudge DOWN, else UP.
    """
    if not ctx.get("is_captured_regime"):
        return RuleResult(prob, False, "")
    gap = ctx.get("poll_house_lean")
    if gap is None or float(gap) <= CAPTURED_POLL_GAP_PP:
        return RuleResult(prob, False, "")
    # Scale the shift with the gap, capped; direction set by who YES backs.
    magnitude = min(CAPTURED_POLL_MAX_SHIFT, (float(gap) - CAPTURED_POLL_GAP_PP) * 0.01)
    toward_frontrunner = bool(ctx.get("subject_is_frontrunner", False))
    new = prob - magnitude if toward_frontrunner else prob + magnitude
    return RuleResult(
        _clamp(new),
        True,
        f"captured-polling: independents favor challenger by {float(gap):.1f}pp "
        f"(>{CAPTURED_POLL_GAP_PP:.0f}pp) -> tilt {magnitude:+.2f} "
        f"{'down (YES=frontrunner)' if toward_frontrunner else 'up (YES=challenger)'}",
    )


def f_wrong_electorate(prob: float, ctx: dict[str, Any]) -> RuleResult:
    """Rule A (Tier-1) - Wrong-electorate flag. HIGHEST RECALL (every leadership/runoff).

    The resolver is a SELECTORATE (party leadership / PM vote), a RUNOFF, or a
    BY-ELECTION - NOT the polled general public. Selectorates skew activist; runoffs
    consolidate against a broad-but-shallow frontrunner. The crowd anchors the
    general-population poll.

    TEMPERED (live test): the naive reflex "fade the public-poll frontrunner" is a
    TRAP - consolidation often runs WITH the leader. So this rule ONLY fires when an
    explicit `consolidation_against_leader` signal is present, and even then applies
    a modest fade. Absent that signal it merely FLAGS the wrong-electorate mismatch
    (lowering confidence) without moving the probability.

    TODO(WP2/parse): derive `resolver_type` from market metadata/resolution_criteria
    and `consolidation_against_leader` from eliminated-candidate endorsements /
    second-preference polling. Until then this is a confidence-only flag.
    """
    resolver = ctx.get("resolver_type", "general")
    if resolver not in ("selectorate", "runoff", "by_election"):
        return RuleResult(prob, False, "")
    if not ctx.get("subject_is_frontrunner", False):
        # YES doesn't back the public frontrunner -> nothing to fade.
        return RuleResult(prob, False, f"wrong-electorate ({resolver}) but YES is not frontrunner")
    against = ctx.get("consolidation_against_leader")
    if against is None:
        # Mismatch noted, but direction unknown - do NOT move the prob (trap guard).
        return RuleResult(
            prob,
            False,
            f"wrong-electorate FLAG ({resolver}): selectorate/runoff != polled public, "
            "but consolidation direction unknown -> no move, lower confidence",
        )
    if not bool(against):
        return RuleResult(
            prob, False, f"wrong-electorate ({resolver}): consolidation runs WITH leader"
        )
    # Consolidation confirmed AGAINST the frontrunner -> modest fade of YES.
    new = prob - min(0.10, (prob - 0.5) * 0.4) if prob > 0.5 else prob
    return RuleResult(
        _clamp(new),
        new != prob,
        f"wrong-electorate ({resolver}) + consolidation AGAINST frontrunner -> fade YES "
        f"{new - prob:+.2f}",
    )


def f_vote_seat_divergence(prob: float, ctx: dict[str, Any]) -> RuleResult:
    """Rule C (Tier-1) - Vote-tie / seat-price divergence. The ONE live-confirmed edge.

    In FPTP/mixed systems, seats are a STEP function of a near-zero vote gap. When
    the vote lead is INSIDE the margin of error but the seat-leader is priced as a
    lopsided winner (>0.65), the crowd has over-converted a coin-flip into a near-
    certainty. Fade the MAGNITUDE toward ~0.6 (NOT a contrarian flip - the leader is
    still favored, just not by that much). Caught West Bengal; live Makerfield/Peru.

    Fires only on FPTP, within-MoE, and a confident YES (>RULE_C_FADE_ANCHOR).
    """
    if not ctx.get("is_fptp", False):
        return RuleResult(prob, False, "")
    lead = ctx.get("vote_lead_pp")
    if lead is None:
        return RuleResult(prob, False, "")
    moe = float(ctx.get("vote_moe_pp", DEFAULT_MOE_PP))
    if abs(float(lead)) >= moe:
        return RuleResult(prob, False, f"FPTP but vote lead {float(lead):.1f}pp >= MoE {moe:.1f}pp")
    if prob <= RULE_C_FADE_ANCHOR:
        return RuleResult(prob, False, f"FPTP within-MoE but YES already <= {RULE_C_FADE_ANCHOR}")
    # Magnitude fade toward the anchor (never below it; never flips the favorite).
    new = RULE_C_FADE_ANCHOR
    return RuleResult(
        _clamp(new),
        True,
        f"vote/seat divergence: vote lead {abs(float(lead)):.1f}pp < MoE {moe:.1f}pp but "
        f"seat-leader priced {prob:.2f} -> magnitude fade to {new:.2f}",
    )


# Priority order: most-precise / live-confirmed first. Rules compose: each takes the
# running probability from the previous one.
_RULES = (f_vote_seat_divergence, f_captured_poll, f_wrong_electorate)


@register_forecaster("domain_rules")
class DomainRulesForecaster:
    """Deterministic Tier-1 electoral through-lines. Plugin key: 'domain_rules'.

    Reads parsed features from `market.metadata["features"]` (a dict) when present.
    Runs each rule in priority order, composing adjustments to a 0.5 base. If no
    rule fires (the default until connectors emit features), returns the base rate
    at low confidence - a safe degrade-to-baseline.
    """

    key = "domain_rules"

    def predict(self, market: Market, context: str) -> Prediction:
        features = market.metadata.get("features", {})
        if not isinstance(features, dict):
            features = {}
        # Base prob: the rules fade/tilt a PRIOR P(YES), not the crowd price. Seed it
        # from an optional reference-class `prior_yes` feature (e.g. a poll-derived
        # seat-leader prior); 0.5 is the documented null when absent. This is NOT the
        # market price - the forecaster stays price-free by contract.
        try:
            prob = _clamp(float(features.get("prior_yes", 0.5)))
        except (TypeError, ValueError):
            prob = 0.5
        notes: list[str] = []
        fired_any = False
        for rule in _RULES:
            result = rule(prob, features)
            prob = result.prob
            if result.note:
                notes.append(result.note)
            fired_any = fired_any or result.fired

        if not fired_any:
            reason = "no Tier-1 rule fired"
            if notes:
                reason += " (" + "; ".join(notes) + ")"
            else:
                reason += " - no parsed features in market.metadata['features']"
            return Prediction(prob_yes=0.5, confidence="low", reasoning=f"domain_rules: {reason}")

        confidence: Confidence = "medium"
        return Prediction(
            prob_yes=_clamp(prob),
            confidence=confidence,
            reasoning="domain_rules: " + "; ".join(notes),
        )
