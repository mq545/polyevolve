"""Baseline forecaster - the trivial reference implementation.

Returns a flat P(YES)=0.5 regardless of input. It earns no edge and is not meant
to: it exists to make the Forecaster contract concrete (copy this file to start a
real one) and to serve as the null control any real forecaster must beat in the
ledger. Self-registers as @register_forecaster("baseline").
"""

from __future__ import annotations

from polyevolve.core.registry import register_forecaster
from polyevolve.core.types import Market, Prediction


@register_forecaster("baseline")
class BaselineForecaster:
    """Flat 0.5 base-rate stub. Plugin key: 'baseline'."""

    key = "baseline"

    def predict(self, market: Market, context: str) -> Prediction:
        return Prediction(prob_yes=0.5, confidence="low", reasoning="base-rate stub")
