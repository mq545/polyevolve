# Platform Build Plan - extensible, shareable market-experiment kit

Goal: a friend clones the repo, writes ~20 lines, and runs their own experiment whose
results land in the shared forward ledger. Markets, research connectors, and forecasters
are all **plugins** discovered by a registry - adding one never touches core code.
Implements the funnel in [ARCHITECTURE.md](ARCHITECTURE.md); experiment-first, no live trading.

## Target package layout
```
src/superpod/
  core/
    types.py        # Market, ResearchContext, Prediction, Bet dataclasses (frozen)
    interfaces.py   # Protocols: MarketSource, ResearchConnector, Forecaster
    registry.py     # @register_market / @register_connector / @register_forecaster + get()
  markets/          # MarketSource plugins - one file each, self-registering on import
    polymarket.py · kalshi.py · __init__.py(auto-imports all)
  connectors/       # ResearchConnector plugins - news · polls · pageviews · finmarkets · web
  forecasters/      # Forecaster plugins - llm_ensemble (qwen3 lenses) · domain_rules · baseline
  harness/
    run.py          # pull market → gather connectors → forecaster → score vs crowd → rubric
    rubric.py       # the 8 checks (ARCHITECTURE.md) as pure functions returning pass/flags
  ledger/
    forward_ledger.py   # DONE - paper_bets table + auto-grade (move here from superpod/)
  scout/
    efficiency.py   # category × thinness map (from the live-scan scripts)
  experiments/      # user recipes: a tiny dataclass or YAML per experiment
  cli.py            # `superpod scout|run|ledger` (single entrypoint)
legacy/             # the ~10 one-off scripts, parked read-only for reference, not deleted
CONTRIBUTING.md     # "add your own market / connector / forecaster in 20 lines"
```

## The three plugin interfaces (the contract everything builds against)
```python
# core/interfaces.py  (typing.Protocol - duck-typed, no inheritance required)
class MarketSource(Protocol):
    key: str                                              # "polymarket"
    def list_markets(self, filt: MarketFilter) -> Iterable[Market]: ...
    def get_resolution(self, external_id: str) -> Resolution | None: ...
    def order_book(self, external_id: str) -> OrderBook | None: ...   # for the executability check

class ResearchConnector(Protocol):
    key: str                                              # "polls"
    categories: tuple[str, ...]                           # ("politics",) or ("*",)
    def fetch(self, ctx: ResearchContext) -> dict: ...    # point-in-time, leakage-guarded
    def render(self, payload: dict) -> str: ...           # text for the prompt (or "" = no-data)

class Forecaster(Protocol):
    key: str                                              # "llm_ensemble"
    def predict(self, market: Market, context: str) -> Prediction: ...  # P(YES)+conf+reasoning
```
Registration is one decorator: `@register_market("polymarket")` on the class. The registry
auto-discovers everything under markets/ connectors/ forecasters/ at import. **Core never
imports a plugin; plugins import core.** That's what makes it safe to add one.

## Extension story (what we are optimizing for)
Add a market:        new file in `markets/`, class with the 3 methods, `@register_market("x")`.
Add a connector:     new file in `connectors/`, `fetch`+`render`, `@register_connector("x")`.
Add a forecaster:    new file in `forecasters/`, `predict`, `@register_forecaster("x")`.
Run an experiment:   a recipe naming {market, category filter, connectors[], forecaster,
                     lead_days} → `superpod run my_recipe` → predictions → forward ledger.
CONTRIBUTING.md ships a copy-paste 20-line template for each, plus a `tests/` fixture pattern.

## Migration map (reuse, don't rewrite)
- market_sources/polymarket.py, kalshi.py → markets/ (wrap to the interface; keep logic)
- data_sources/{gdelt_doc,polls,pageviews,finmarkets}.py → connectors/ (add key+categories)
- scripts/kill_test.py forecaster (lenses+trimmed-mean+abstain) → forecasters/llm_ensemble.py
- domain Tier-1 rules → forecasters/domain_rules.py (f_captured_poll, f_wrong_electorate…)
- scripts/{arb_probe,latency_edge,efficiency}.py logic → scout/ + the rubric executability check
- scripts/forward_ledger usage stays; module moves under ledger/
- everything else in scripts/ → legacy/ (kept for reference, not imported)

## Build breakdown (for the agent dev team - file ownership is DISJOINT to avoid conflicts)
- **WP0 Core (foundation, must land first):** core/{types,interfaces,registry}.py + package
  skeleton + a working reference plugin (Polymarket market) + the CLI shell. One coherent unit.
- **WP1 Markets:** migrate Kalshi; add order_book() to Polymarket. Owns markets/* only.
- **WP2 Connectors:** migrate news/polls/pageviews/finmarkets + add a `web` connector. Owns connectors/* only.
- **WP3 Forecasters:** llm_ensemble + domain_rules + baseline(base-rate). Owns forecasters/* only.
- **WP4 Harness+Rubric:** run.py + rubric.py (8 checks as functions). Owns harness/* only.
- **WP5 Scout:** efficiency map. Owns scout/* only.
- **WP6 Integrate+Docs (last):** wire cli.py + __init__ auto-discovery, CONTRIBUTING.md,
  README quickstart, run ruff+mypy+pytest, fix integration. Owns the shared glue.

## Acceptance bar
`uv run ruff check` + `uv run mypy` clean; `uv run pytest` green; `superpod scout` and a
1-market `superpod run` work end-to-end into the ledger; CONTRIBUTING's 20-line examples
actually run. Existing behavior preserved (forecaster reproduces a cached kill_test number).
