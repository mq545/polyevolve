from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class Market:
    venue: str
    external_id: str
    cross_venue_id: str | None
    question: str
    close_time: datetime | None
    status: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Resolution:
    venue: str
    external_id: str
    outcome: str
    resolved_at: datetime


class MarketSource(Protocol):
    name: str

    def list_markets(self, filters: dict[str, Any]) -> Iterable[Market]: ...

    def get_resolution(self, external_id: str) -> Resolution | None: ...
