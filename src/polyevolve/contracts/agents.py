from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .markets import Market


@dataclass(frozen=True)
class Prediction:
    market_venue: str
    market_external_id: str
    agent_name: str
    model_name: str
    probability_yes: float
    confidence: float
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    uncertainty_drivers: list[str] = field(default_factory=list)
    data_sources_used: list[str] = field(default_factory=list)
    market_price_at_prediction: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ResearchAgent(Protocol):
    name: str
    domain: str

    def predict(self, market: Market, data: dict[str, Any]) -> Prediction: ...
