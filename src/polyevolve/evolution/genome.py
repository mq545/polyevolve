"""Genome - the evolvable forecasting strategy.

A Genome fully determines how the agent predicts: the prompt text, which data
sources feed it (and weights), and inference config. Two identical genomes
produce identical predictions on identical inputs, so the genome_hash is a sound
cache key. The hash is deterministic (sorted, md5) - NOT Python's salted hash.

This mirrors the v2/v3 genome design (see project_polymarket_evolution memory),
trimmed to what v0 actually has wired (prompt + data weights + effort). Model
selection is a separate axis (the evaluator takes a model), so it's not in the
genome's identity - a model sweep reuses one genome across models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Genome:
    system_prompt: str
    domain_context: str
    # data source name -> weight; weight 0 disables. Rendered context for a
    # source is included only if its weight > 0; weights also scale truncation.
    data_weights: dict[str, float] = field(default_factory=lambda: {"gdelt_news": 1.0})
    effort: str = "medium"  # low | medium | high (used by Anthropic path)
    max_context_chars: int = 8000

    def hash(self) -> str:
        """Deterministic content hash; the cache key for predictions."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.md5(payload).hexdigest()  # noqa: S324 - non-crypto cache key

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genome:
        return cls(
            system_prompt=d["system_prompt"],
            domain_context=d["domain_context"],
            data_weights=dict(d.get("data_weights", {"gdelt_news": 1.0})),
            effort=d.get("effort", "medium"),
            max_context_chars=int(d.get("max_context_chars", 8000)),
        )


def default_genome() -> Genome:
    """The v0 baseline genome - current production prompt + config."""
    from polyevolve.agents.foreign_politics_agent import (
        DOMAIN_CONTEXT_FOREIGN_POLITICS,
        SYSTEM_PROMPT,
    )

    return Genome(
        system_prompt=SYSTEM_PROMPT,
        domain_context=DOMAIN_CONTEXT_FOREIGN_POLITICS,
    )
