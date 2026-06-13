<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-wordmark-dark.png">
    <img src="assets/logo-wordmark.png" alt="PolyEvolve" width="420">
  </picture>
</p>

<p align="center"><em>Evolve trading strategies for prediction markets — and measure the edge without fooling yourself.</em></p>

---

**PolyEvolve** turns a forecasting/trading strategy into a *genome* and **evolves** it
against real, resolved Polymarket & Kalshi markets — optimizing for calibration or
net-of-spread return. Three things make it worth your time:

- **🧬 Evolve, don't hand-tune.** `polyevolve evolve` searches strategy-space (prompts,
  ensembling, calibration, sizing, abstention) and reports the champion vs the seed on a
  **held-out** split. One line: `pe.evolve(markets, objective="return")`.
- **🔌 Plug in your own.** Markets, research connectors, and forecasters are all **plugins** —
  ~20 lines + one decorator and the registry auto-discovers it. Your strategy runs through the
  same harness as everything else.
- **🛡️ An honest harness.** Every result crosses the spread in an adversarial sim, clusters
  correlated markets into events, and is graded **forward** in a paper-bet ledger — so a
  backtest can't lie to you (see [ARCHITECTURE.md](ARCHITECTURE.md) for the 8-check rubric).

Venue-agnostic (Polymarket + Kalshi). Paper predictions only — no live trading.

## Quickstart

```bash
git clone <repo> && cd polymarket-agents
uv sync                                    # installs deps + the `polyevolve` CLI
docker compose up -d postgres              # Postgres (ledger + market dataset)

# 1. SCOUT — where is the crowd thin right now? (live, no model needed)
uv run polyevolve scout

# 2. BUILD a dataset of resolved markets to evolve against ($0, any domain)
uv run polyevolve snapshot --set demo --domain all --min-volume 10000 \
    --no-research --limit 200

# 3. EVOLVE a strategy — watch seed -> champion on a held-out split
uv run polyevolve evolve --snapshot-set demo --objective return
```

`polyevolve evolve` runs the strategy genome with a local LLM (Ollama qwen3 by default — see
[Model routing](#model-routing)) and prints the evolved champion's knobs and its holdout lift.
Prefer code? The same loop is **six composable verbs**:

```python
import polyevolve.api as pe
qs   = pe.markets(source="polymarket", snapshot_set="demo")   # resolved markets
pools = pe.gather(qs)                                          # leakage-safe evidence
best = pe.evolve(qs, pools, objective="return")               # search strategy-space
print(best.knobs, best.val_fitness)                           # champion + its holdout score
```

Adding your own forecaster/connector/market is the whole point — see
[Add your own](#add-your-own-the-whole-point).

## Add your own (the whole point)

Adding a market, a research connector, or a forecaster is one file + one
decorator - core never imports your plugin, the registry auto-discovers it. See
**[CONTRIBUTING.md](CONTRIBUTING.md)** for copy-paste 20-line templates of each,
plus the no-LLM test pattern. Run yours with
`uv run polyevolve run --forecaster <your_key> ...`.

Confirm discovery sees everything on disk:

```bash
uv run python -c "from polyevolve.core import registry; registry.discover(); \
  print(sorted(registry.all_markets()), sorted(registry.all_connectors()), \
  sorted(registry.all_forecasters()))"
```

## How scoring works

Every experiment runs through one harness and is scored on net-of-spread terms, never a
mid-quote: forecasts go through the adversarial trading sim (`polyevolve.bench.sim` - crosses
the spread, walks the order book, shares a per-EVENT Kelly budget, computes significance on
events not markets), and calibration is measured with proper reliability/resolution
decomposition (`polyevolve.bench.scoring`). Belief in any result is earned only by the forward
paper-bet ledger, graded against reality - see [ARCHITECTURE.md](ARCHITECTURE.md) for the
exploration -> confirmation funnel and the 8-check rubric.

## Stack

- Python 3.12 (managed by `uv`)
- PostgreSQL 16 (Docker)
- Anthropic SDK (direct, for production calibration runs)
- LiteLLM (for local Ollama / OpenAI / multi-provider dev)
- httpx, pydantic, apscheduler

## Setup

```bash
# Install dependencies and create venv
uv sync

# Copy env template and fill in keys
cp .env.example .env
$EDITOR .env

# Start Postgres
docker compose up -d postgres
```

The day-to-day entrypoint is the `polyevolve` CLI (see the Quickstart above). The
older `polyevolve.orchestration.daily_run` / `evaluate` modules are the legacy,
pre-plugin foreign-politics pipeline and the offline snapshot/backtest path; they
remain for the evolution experiments but new work goes through `polyevolve run`.

## Observability

Every LLM call is traced to the Postgres `llm_calls` table - prompt, response,
tokens, cache stats, latency, and estimated cost. Inspect the pipeline with the
CLI (no SQL required):

```bash
uv run polyevolve predictions --limit 25   # recent predictions
uv run polyevolve calibration              # decile calibration + edge vs market
uv run polyevolve cost                     # token usage + $ cost
uv run polyevolve runs                     # per-day run summary
uv run polyevolve coverage                 # markets tracked / predicted / resolved
uv run polyevolve traces --limit 25        # recent LLM calls
uv run polyevolve traces --market 597964   # calls for one market
```

For ad-hoc SQL, install a Postgres client in WSL and connect to the published
port:

```bash
sudo apt install -y postgresql-client
psql postgresql://superpod:superpod@localhost:5432/superpod
# then: SELECT * FROM v_calibration_vs_market;
```

Inspection views: `v_recent_predictions`, `v_run_summary`, `v_cost`,
`v_market_coverage`, `calibration`, `v_calibration_vs_market`.

**Langfuse (optional LLM tracing UI):** off by default. Postgres tracing covers
the SQL/CLI path; Langfuse adds a browser UI with per-call traces and
calibration scoring. Enable by setting `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
(+ `uv add langfuse`) - see `.env.example`. Cloud free tier is the fastest start;
self-hosting is a multi-container stack, deferred until needed.

## Architecture

The package (`src/polyevolve/`) is **two cooperating subsystems**: a clean experiment
*surface* and the operational *pipeline* that feeds it. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the exploration → confirmation funnel.

**1. The PolyEvolve surface** (what you compose and share) - six plain-value verbs in
[`polyevolve.api`](src/polyevolve/api.py):

```python
import polyevolve.api as pe
qs    = pe.markets(source="polymarket")        # resolved questions (+ price, outcome)
pools = pe.gather(qs)                           # frozen, leakage-safe evidence
g     = pe.seed(use_ensemble=True)              # a genome (Question, Pool) -> Forecast
score = pe.score(g, qs, pools, objective="calibration")
best  = pe.evolve(qs, pools, objective="return")
fc    = pe.forecast(best.genome, qs[0], pools[0])
```

| Package | Role |
|---|---|
| `core/`, `contracts/` | the registry + Protocol types (`Model`, `MarketSource`, …) |
| `markets/`, `connectors/`, `forecasters/` | **the plugins** `discover()` auto-registers |
| `data_sources/` | the raw fetchers (GDELT, polls, trends, Manifold, World Bank, …) that connectors wrap |
| `reason/` | the genome: typed nodes, the seed scaffold, joint inference |
| `bench/` | scoring, calibration, the net-of-spread return sim (`sim.py`), datasets |
| `evolve/` | the knob-space optimizer behind `pe.evolve` |
| `harness/` | `run_experiment` → the 8-check `rubric` |
| `ledger/` | the forward paper-bet ledger (the un-foolable gate) |
| `models/` | `build_model` (Anthropic direct / LiteLLM) + `coerce_rows` |
| `scout/`, `observability/`, `storage/` | efficiency map · LLM tracing · Postgres |

**2. The offline pipeline** (builds the data the surface reads): `orchestration/snapshot.py`
populates the Postgres `eval_snapshots` table via the legacy `market_sources/` client and
`data_sources/registry`; `evolution/` + `agents/` are its backtest-fitness machinery. New
work goes through the surface above; these remain because they own data ingestion.

## Model routing

`DEFAULT_MODEL` decides which backend serves predictions:

| `DEFAULT_MODEL` value | Backend | Notes |
|---|---|---|
| `claude-sonnet-4-6` | Anthropic SDK (direct) | Default. Full feature set: prompt caching + adaptive thinking. |
| `claude-opus-4-7`, `claude-haiku-4-5` | Anthropic SDK (direct) | Same path. |
| `ollama/qwen3:30b-a3b-instruct-2507-q4_K_M` (or any `ollama/...`) | LiteLLM → local Ollama | For dev iteration on the 3090. No caching. |
| `openai/gpt-4o` (or any `openai/...`) | LiteLLM → OpenAI | If you ever want a cross-provider comparison. |

Any model ID containing `/` routes through LiteLLM; bare Claude IDs go direct.

### Running locally on Ollama

1. Install Ollama: https://ollama.com/download (Windows native or WSL)
2. Pull an instruct model with strong structured-output support:
   ```bash
   ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M   # what we run (~18GB VRAM)
   ollama pull qwen2.5:14b                           # lighter alternative (~9GB VRAM)
   ```
3. Set in `.env`:
   ```
   DEFAULT_MODEL=ollama/qwen3:30b-a3b-instruct-2507-q4_K_M
   ```
4. Run normally - `uv run polyevolve run --forecaster baseline ...` (the local model
   serves any LLM forecaster; the `baseline` forecaster needs no model at all).

Caveat: local models calibrate worse than Claude on probabilistic forecasting. Use the local path for plumbing tests and dev iteration; switch back to `claude-sonnet-4-6` for the actual calibration measurement that decides whether the thesis works.

## Roadmap

- **v0** - single agent, foreign politics, text-only data, paper predictions, calibration measurement
- **v1** - data ingestion + calibration correction layer + hardcoded risk gate
- **v2** - multi-agent ensemble + offline evolutionary loop
- **v3** - live evolutionary loop with hard guardrails (kill switches, holdout requirements)
- **v4+** - execution layer (Kalshi US-legal; Polymarket non-US only)
