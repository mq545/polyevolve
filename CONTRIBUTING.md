# Contributing - add your own market, connector, or forecaster

This platform is built to be extended by **writing one small file**. Markets,
research connectors, and forecasters are all *plugins*: a class with a couple of
methods and one registration decorator. The registry auto-discovers every module
under `markets/`, `connectors/`, and `forecasters/` at import time - so adding a
plugin **never touches core code**. Core never imports a plugin; plugins import
core. That one-way dependency is what makes adding one safe.

Run an experiment with your plugin:

```bash
uv run polyevolve run --market polymarket --forecaster <your_key> \
    --connectors <your_connector_key> --category politics --lead-days 30 --limit 5
```

Each forecast is scored against the crowd, run through the 8-check rubric, and
(unless `--no-ledger`) logged into the forward paper-bet ledger, which grades it
against reality at resolution. That ledger is the only thing that earns belief.

The three contracts live in `src/polyevolve/core/interfaces.py`; the value types
(`Market`, `Prediction`, …) in `src/polyevolve/core/types.py`. Copy a template
below, drop it in the right directory, and you are done - no registration list to
edit, no import to add anywhere.

---

## Add a forecaster

A forecaster turns `(market, rendered research text)` into a `P(YES)`. It is
**never shown the market price** - only the question, resolution criteria, and
research. Drop this in `src/polyevolve/forecasters/my_forecaster.py`:

```python
from __future__ import annotations

from polyevolve.core.registry import register_forecaster
from polyevolve.core.types import Market, Prediction


@register_forecaster("my_forecaster")          # <- your unique key
class MyForecaster:
    key = "my_forecaster"

    def predict(self, market: Market, context: str) -> Prediction:
        # `context` is the assembled, price-free research block (may be "").
        # Return a calibrated P(YES), a confidence band, and your real reasoning.
        prob = 0.5  # ... your logic here ...
        return Prediction(prob_yes=prob, confidence="low", reasoning="why")
```

Test it with the null-control baseline first: `--forecaster baseline` always
returns 0.5; your forecaster has to beat it in the ledger to mean anything.

---

## Add a research connector

A connector supplies **point-in-time, price-free** research text for the prompt.
`fetch` returns a leakage-guarded payload (only data strictly before `ctx.as_of`);
`render` turns it into prompt text, or `""` for no-data (so the forecaster sees
the gap as absence, not noise). `categories` declares which market categories it
applies to; `("*",)` means all. Drop this in
`src/polyevolve/connectors/my_connector.py`:

```python
from __future__ import annotations

from typing import Any

from polyevolve.core.registry import register_connector
from polyevolve.core.types import ResearchContext


@register_connector("my_connector")            # <- your unique key
class MyConnector:
    key = "my_connector"
    categories: tuple[str, ...] = ("politics",)  # or ("*",) for every category

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]:
        # ONLY return data dated strictly before ctx.as_of (no leakage!).
        # ctx gives you: question, as_of, tags, category, and the full market.
        return {"finding": f"something about {ctx.question}"}

    def render(self, payload: dict[str, Any]) -> str:
        return str(payload.get("finding", ""))  # "" == no data this time
```

---

## Add a market source

A market source supplies normalized markets, their resolutions, and order books.
`list_markets` yields `Market`s; `get_resolution`/`order_book` return `None`
(never raise) when the venue has no answer. Drop this in
`src/polyevolve/markets/my_market.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

from polyevolve.core.registry import register_market
from polyevolve.core.types import Market, MarketFilter, OrderBook, Resolution


@register_market("my_market")                  # <- your unique key
class MyMarket:
    key = "my_market"

    def list_markets(self, filt: MarketFilter) -> Iterable[Market]:
        # Honor filt (category / tags / open_only / resolves_within_days).
        # Put the venue's crowd price in metadata (e.g. metadata["outcomePrices"]).
        return []

    def get_resolution(self, external_id: str) -> Resolution | None:
        return None                            # None until settled

    def order_book(self, external_id: str) -> OrderBook | None:
        return None                            # for the executability rubric check
```

Real, working references to copy: `markets/polymarket.py` and `markets/kalshi.py`.

---

## Test your plugin (no LLM, no network)

The platform's end-to-end test (`tests/test_harness_e2e.py`) is the pattern to
copy: register a fake plugin, call `registry.discover()`, run it through
`harness.run_experiment(...)` with the **baseline** forecaster, and assert on the
`ExperimentResults`. No model call, no network, no DB. Use it as a template for a
unit test of your own plugin.

## Before you open a PR - the gate

All three must be green (CI runs them):

```bash
uv run ruff check
uv run mypy src/polyevolve
uv run pytest
```

Style: `from __future__ import annotations` at the top, typed signatures
(`mypy --strict` is on), 100-col lines. Keep it **$0** - no paid APIs, and do not
run BigQuery (it is over the free tier).
