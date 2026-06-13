"""Per-model training knowledge cutoff registry.

No API or local metadata reliably exposes a model's knowledge cutoff (verified
2026-05-30: Ollama /api/show has no training fields; Anthropic Models API
exposes created_at but not cutoff; asking the model is unreliable). So we
maintain this hardcoded registry, dated conservatively.

Why this matters: a backtest is only honest if the model didn't already learn
the outcome during training. `is_clean_for_backtest(model, resolved_at)` answers
"could this model know this result from training?" - used to exclude
contaminated markets from backtest calibration.

Contamination error is ASYMMETRIC: wrongly believing a market is clean (when the
model actually knows the answer) inflates apparent skill - the costly error. So:
- date cutoffs conservatively LATE when unsure, and
- require a safety margin after the cutoff before trusting a market as clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class Cutoff:
    date: datetime
    confidence: str  # "official" | "inferred" | "unknown"
    source: str


# Conservative safety margin: even past the stated cutoff, training data ingests
# events with a lag, so treat the cutoff + this margin as the true clean boundary.
SAFETY_MARGIN = timedelta(days=90)


_REGISTRY: dict[str, Cutoff] = {
    # qwen2.5: NO official cutoff published. Community-inferred ~Oct 2023
    # (QwenLM/Qwen3 discussion #1093; HaoooWang/llm-knowledge-cutoff-dates).
    # Dated late on purpose given the asymmetric risk.
    "ollama/qwen2.5:14b": Cutoff(
        date=datetime(2023, 12, 31, tzinfo=UTC),
        confidence="inferred",
        source="community-inferred ~Oct 2023; dated to year-end conservatively",
    ),
    # qwen3 *-2507 line (released ~Jul 2025). No official cutoff published; the
    # 2507 tag implies training data through roughly early-mid 2025. Dated late
    # and conservative given the asymmetric contamination risk. Same entry covers
    # both the thinking and instruct variants (get_cutoff strips the :tag).
    "ollama/qwen3:30b-a3b-thinking-2507-q4_K_M": Cutoff(
        date=datetime(2025, 4, 30, tzinfo=UTC),
        confidence="inferred",
        source="no official cutoff; 2507 release => inferred ~early-mid 2025, dated late",
    ),
    "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M": Cutoff(
        date=datetime(2025, 4, 30, tzinfo=UTC),
        confidence="inferred",
        source="no official cutoff; 2507 release => inferred ~early-mid 2025, dated late",
    ),
    # Anthropic publishes a "reliable knowledge cutoff" in docs (not via API).
    # Update from the model card when migrating models.
    "claude-sonnet-4-6": Cutoff(
        date=datetime(2025, 1, 1, tzinfo=UTC),
        confidence="official",
        source="Anthropic model docs reliable-cutoff (verify on model change)",
    ),
    "claude-opus-4-7": Cutoff(
        date=datetime(2025, 1, 1, tzinfo=UTC),
        confidence="official",
        source="Anthropic model docs reliable-cutoff (verify on model change)",
    ),
}


def get_cutoff(model_name: str) -> Cutoff | None:
    """Look up a model's cutoff. Tries exact id, then the base before any tag."""
    if model_name in _REGISTRY:
        return _REGISTRY[model_name]
    # tolerate version/tag drift, e.g. "ollama/qwen2.5:14b-instruct-q4"
    base = model_name.split(":", 1)[0]
    for key, cutoff in _REGISTRY.items():
        if key.split(":", 1)[0] == base:
            return cutoff
    return None


def is_clean_for_backtest(model_name: str, resolved_at: datetime) -> bool:
    """True if a market resolving at `resolved_at` is safe to backtest on `model`.

    Safe = resolution happened after the model's cutoff + safety margin, so the
    outcome cannot have been in training data. Unknown model => not clean (fail
    closed: never claim cleanliness we can't justify).
    """
    cutoff = get_cutoff(model_name)
    if cutoff is None:
        return False
    boundary = cutoff.date + SAFETY_MARGIN
    resolved = resolved_at if resolved_at.tzinfo else resolved_at.replace(tzinfo=UTC)
    return resolved > boundary
