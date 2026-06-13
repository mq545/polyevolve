"""Pollster lean map - the symbolic half of the captured-polling reweight.

The crowd misprices elections where the SALIENT signal (the poll average) is captured: in some
countries a bloc of government-aligned pollsters systematically over-states the incumbent, so a
naive average points the wrong way (Hungary 2026: Fidesz-aligned houses favored Fidesz; Tisza
won). This module flags pollsters by lean so a reweight node can DROP the captured ones and let
the forecaster read only the independent polls.

CURATED + EXTENSIBLE (and intentionally explicit): leans are sourced from public reporting and,
for Hungary, the Wikipedia "alleged bias / conflict of interest" table the polls extractor
surfaces. This is a *named, auditable* prior, not a hidden one - add countries by appending here.
Matching is case-insensitive substring on the pollster name.
"""

from __future__ import annotations

__all__ = ["GOV_ALIGNED", "OPP_OR_INDEPENDENT", "pollster_lean"]

# Government/incumbent-aligned ("captured") pollster name fragments, by context.
GOV_ALIGNED: dict[str, tuple[str, ...]] = {
    # Hungary - Fidesz-aligned houses (the captured bloc).
    "hungary": (
        "nézőpont",
        "nezopont",
        "századvég",
        "szazadveg",
        "alapjogokért",
        "alapjogokert",
        "xxi. század",
        "xxi. szazad",
        "21 század",
        "real-pr",
        "ravasz",
    ),
}

# Independent / opposition-leaning houses (kept, and useful for sanity).
OPP_OR_INDEPENDENT: dict[str, tuple[str, ...]] = {
    "hungary": (
        "medián",
        "median",
        "republikon",
        "idea",
        "závecz",
        "zavecz",
        "publicus",
        "iránytű",
        "iranytu",
        "21 kutatóközpont",
        "21 kutatokozpont",
        "zri",
    ),
}


def _context_for(text: str) -> str | None:
    t = text.lower()
    if any(k in t for k in ("hungar", "magyar", "tisza", "fidesz", "orbán", "orban")):
        return "hungary"
    return None


def pollster_lean(pollster: str, *, context_hint: str = "") -> str:
    """Classify a pollster as 'gov' (captured), 'ind' (independent/opposition), or 'unknown'.

    ``context_hint`` (e.g. the question text or country) selects the country map; if empty we
    scan all contexts. Substring, case-insensitive.
    """
    name = pollster.lower()
    ctx = _context_for(context_hint) if context_hint else None
    contexts = [ctx] if ctx else list(GOV_ALIGNED)
    for c in contexts:
        if any(frag in name for frag in GOV_ALIGNED.get(c, ())):
            return "gov"
    for c in contexts:
        if any(frag in name for frag in OPP_OR_INDEPENDENT.get(c, ())):
            return "ind"
    return "unknown"
