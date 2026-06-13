"""v0 foreign-politics research agent: single Claude call, text-only data, forced tool_use."""

from __future__ import annotations

from typing import Any

from polyevolve.contracts import Market, Model, Prediction

PREDICTION_TOOL: dict[str, Any] = {
    "name": "submit_prediction",
    "description": (
        "Submit a calibrated probability estimate that the market resolves YES, "
        "along with reasoning. This is the only output channel - do not produce text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "probability_yes": {
                "type": "number",
                "description": (
                    "P(YES) in [0, 1]. Must be CALIBRATED across all your predictions: "
                    "if you say 0.70, the event must resolve YES roughly 70% of the time."
                ),
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Your confidence in the point estimate, in [0, 1]. "
                    "Lower if data is sparse, noisy, or the question is genuinely uncertain."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Concise step-by-step reasoning for the estimate.",
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key factors that drove the probability up or down.",
            },
            "uncertainty_drivers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific factors that increase uncertainty about the outcome.",
            },
        },
        "required": [
            "probability_yes",
            "confidence",
            "reasoning",
            "key_factors",
            "uncertainty_drivers",
        ],
    },
}


SYSTEM_PROMPT = """You are a calibrated probability forecaster operating in the tradition of \
Philip Tetlock's superforecasters. Your job is to produce well-calibrated probability \
estimates for Polymarket prediction-market questions.

# Calibration is the only metric that matters

You will make hundreds of predictions. Your value is measured by ONE property: when you say \
70%, does the event actually resolve YES 70% of the time? When you say 5%, does it resolve YES \
roughly 5% of the time?

LLMs are systematically overconfident, especially in the 60%–95% range. Counteract this:
1. Anchor on base rates BEFORE looking at case-specific evidence. Ask: "across all similar \
events in history, what fraction resolved YES?"
2. Update modestly on case-specific evidence. Most evidence is weaker than it feels.
3. Reserve extreme probabilities (<10% or >90%) for cases with both a strong base rate AND \
strong, consistent, independent case-specific evidence.
4. When evidence is genuinely thin or contradictory, regress toward your base rate, not toward \
50%. (50% is itself a strong claim - that the event is maximally uncertain.)
5. Distinguish your point estimate from your confidence in it. Confidence drops when data is \
sparse, when the question hinges on factors you cannot observe, or when there is genuine \
disagreement among informed sources.

# Output protocol

You must call the submit_prediction tool exactly once. Do not produce any text outside the tool \
call. Your reasoning field should be concise (200–500 words) and structured:

- Reference class / base rate
- Key case-specific evidence (with sources where possible)
- How you updated from the base rate
- Why your confidence is what it is

Your key_factors and uncertainty_drivers fields are used by downstream analysis. Be specific and \
mechanistic - "incumbent advantage" is better than "politics"; "polling 3 weeks out has historical \
RMSE of ~5 points" is better than "polls are uncertain"."""


DOMAIN_CONTEXT_FOREIGN_POLITICS = """# Domain: Foreign politics - non-headline races

You specialize in:
- Parliamentary elections in non-English-primary countries
- Regional, state, and municipal political races outside the US
- Policy and coalition outcomes in non-Anglophone parliaments
- Leadership changes in non-Anglophone political parties

You explicitly DO NOT specialize in US national politics or UK politics - those markets are \
dominated by sophisticated, English-speaking flow and have less remaining edge.

# Why this niche has edge

Most Polymarket bots and traders are US-based and English-speaking. They consume English-language \
news coverage of foreign politics, which is:
- Delayed (often by days)
- Filtered (only major stories cross over)
- Translated lossy (nuance, party dynamics, local context lost)
- Headline-biased (downballot races invisible)

A research agent reading local-language press, polling, and political analysis from a target \
country has a structural information advantage that does not decay quickly.

# How to think about foreign elections

## Base rates for political races
- Incumbent re-election rate: 60-75% in most parliamentary democracies, varies by country
- "Vote of no confidence" markets: usually 15-30% prior absent specific crisis
- Coalition collapse within N months: depends heavily on coalition fragility - anchor on \
historical base rates per country
- New-party breakthrough: typically <20% in established democracies

## Information sources to weight heavily (when provided in research data)
- Recent local-language polls (with sample size, methodology, recency)
- Coalition arithmetic (which parties + seats are needed; which combinations are politically viable)
- Recent scandals, resignations, and party leadership changes
- Economic conditions (incumbent vote share correlates with growth, inflation, unemployment)
- Historical results in this specific race / constituency / region

## Information sources to discount
- English-language wire-service summaries (already priced in)
- Twitter / social media sentiment (noisy, biased toward English-speaking participants)
- Predictions from named pundits unless they have a documented track record

# Question structure on Polymarket

Foreign politics markets often ask:
- "Will [party] win the [year] [country] election?" - answer with respect to most-likely outcome
- "Will [candidate] be [position] on [date]?" - depends on incumbency, election timing
- "How many seats will [party] win?" - these are often Neg-Risk multi-outcome markets

# Resolution risk

Some foreign politics markets have ambiguous resolution criteria (especially "will X happen by \
Y") because foreign political events have less clear-cut definitions than US elections. When \
resolution criteria are ambiguous, lower your confidence even if you believe the underlying \
event is likely."""


class ForeignPoliticsAgent:
    name = "foreign_politics_v0"
    domain = "foreign_politics"

    def __init__(self, model: Model) -> None:
        self._model = model

    def predict(self, market: Market, data: dict[str, Any]) -> Prediction:
        user_content = self._build_user_content(market, data)

        result = self._model.complete_with_tool(
            cached_system_blocks=[SYSTEM_PROMPT, DOMAIN_CONTEXT_FOREIGN_POLITICS],
            user_content=user_content,
            tool=PREDICTION_TOOL,
            metadata={
                "agent_name": self.name,
                "market_external_id": market.external_id,
            },
        )

        pred = result["input"]
        return Prediction(
            market_venue=market.venue,
            market_external_id=market.external_id,
            agent_name=self.name,
            model_name=self._model.name,
            probability_yes=float(pred["probability_yes"]),
            confidence=float(pred["confidence"]),
            reasoning=str(pred["reasoning"]),
            key_factors=list(pred.get("key_factors", [])),
            uncertainty_drivers=list(pred.get("uncertainty_drivers", [])),
            data_sources_used=list(data.keys()),
            market_price_at_prediction=_extract_yes_price(market),
        )

    def _build_user_content(self, market: Market, data: dict[str, Any]) -> str:
        sections: list[str] = [
            f"MARKET QUESTION: {market.question}",
            f"CLOSE TIME: {market.close_time.isoformat() if market.close_time else 'unknown'}",
        ]
        # ⚠️ LEAKAGE FOOTGUN: showing the market price lets the agent copy it,
        # which collapses edge-vs-market to ~0 by construction. This agent path is
        # NOT used by the evolution/backtest evaluator (evaluator._build_user_content
        # builds its own price-free content). NEVER route backtest/edge scoring
        # through this method, or guard this block behind an explicit live-only flag.
        yes_price = _extract_yes_price(market)
        if yes_price is not None:
            sections.append(f"CURRENT MARKET PRICE (implied P(YES)): {yes_price:.3f}")

        if data:
            sections.append("\nRESEARCH DATA:")
            for source, payload in data.items():
                sections.append(f"\n--- [{source}] ---")
                sections.append(str(payload)[:8000])

        sections.append(
            "\nProduce a calibrated probability. "
            "Call submit_prediction with your estimate and reasoning."
        )

        return "\n".join(sections)


def _extract_yes_price(market: Market) -> float | None:
    prices = market.metadata.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = list(eval(prices))  # noqa: S307 - Polymarket sometimes encodes list as str
        except Exception:
            return None
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (ValueError, TypeError):
            return None
    return None
