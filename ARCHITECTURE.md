# Target Architecture - agentic market-experiment platform

Goal: run **many** $0 experiments across market categories to find a tradeable edge,
**without fooling ourselves**. Belief is earned only by forward, out-of-sample results.

## The shape

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  SOURCES   (pluggable - add a category = add one connector)           │
   │  markets:  Polymarket · Kalshi …      research: news·polls·web·       │
   │                                        esports·pop-culture · …         │
   └───────────────────────────────┬──────────────────────────────────────┘
                                    │
  ══════════ EXPLORATION  (cheap · $0 · unlimited · "fool-yourself" zone) ══════════
                                    │
        ┌───────────────┐   ┌───────▼────────┐
        │  SCOUT        │──►│  HARNESS       │   one path for every experiment:
        │ efficiency map│   │ pull→research→ │   pull a market, gather signal,
        │ thin × edge   │   │ predict→score  │   predict, score vs market
        └───────────────┘   └───────┬────────┘
                                    │
                            ┌───────▼────────┐   power · out-of-sample · executable
                            │  RUBRIC GATE   │   (book-walk) · data-real · edge-type
                            │   (8 checks)   │   · observation-reviewed · cost · fit
                            └───────┬────────┘
                          fail ◄────┤ pass
                         discard    │   (spent $0, belief = 0)
  ═══════════ CONFIRMATION  (the un-foolable gate · forward only) ═══════════
                                    │
                      ┌─────────────▼───────────────┐
                      │  FORWARD PAPER-BET LEDGER    │  log prediction + price NOW,
                      │  auto-grade AT RESOLUTION    │  grade vs reality later,
                      │  calibration + FDR per cat.  │  (un-overfittable, un-fakeable)
                      └─────────────┬───────────────┘
                                    │  forward-confirmed track record
                                    ▼
                            ┌───────────────┐
                            │ BELIEVED EDGE │ → only now is it real
                            └───────────────┘

   AGENTS run across every stage - domain experts (research) + observation-level
   reviewers (read the actual cases, never trust aggregates alone).
```

**The load-bearing idea is the horizontal split.** EXPLORATION (top) is free and
unlimited - throw 100 ideas at the wall, cost stays $0, nothing earns belief.
CONFIRMATION (bottom) is the only thing trusted: an edge is "believed" *only* after a
forward paper-bet track record graded against reality. Forward + out-of-sample can't be
overfit and can't be a mid-quote artifact - the structural defense against false
discovery, the failure mode of "run a bunch of experiments to see what sticks."

## Where the LLM lives (the HARNESS, zoomed in)

```
  INPUTS  (point-in-time · price-FREE)          THE MODEL                    OUTPUTS
  ┌─────────────────────────────────┐      ┌──────────────────────┐    ┌─────────────────────┐
  │ • question + resolution criteria│      │  FORECASTER          │    │ • P(YES) ∈ [0,1]    │
  │ • research context (assembled): │─────►│  (qwen3 local -      │───►│ • confidence        │
  │     news·polls(grounded)·       │      │   SWAPPABLE)         │    │ • REASONING trace   │
  │     pageviews·web·domain-rules  │      │  K reasoning lenses  │    │   (stored, audited) │
  │ • prompt (shift / domain rules) │      │  → trimmed-mean      │    └──────────┬──────────┘
  │ • ✗ NEVER the market price      │      │  → ABSTAIN if no data│               │
  └─────────────────────────────────┘      └──────────────────────┘               │
            ▲ assembled & verified by                  ┌──────────┬──────────┬─────┘
            │ research/domain agents                   ▼          ▼          ▼
   ┌────────┴─────────┐                          score vs mkt  RUBRIC   FORWARD LEDGER
   │ RESEARCH AGENTS  │ (web-enabled)            edge=Δbrier   GATE     (grade later)
   └──────────────────┘                                                      │
            ▲──────────────── fix inputs/rules ◄── OBSERVATION REVIEW AGENT ──┘
                                                    (reads REASONING traces)
```

**Two LLM roles, kept separate:**
- **Model-as-Forecaster** (scored): in = question + assembled research + prompt,
  **price-free + point-in-time**; out = calibrated `P(YES)` + confidence + reasoning
  trace. **Swappable, NOT the bottleneck** (parity was bad inputs, not a weak model).
- **Agents-as-Workforce** (never directly scored): research agents *assemble & verify*
  inputs (real poll tables, pollster bias, candidacy checks); review agents *audit* the
  stored reasoning traces (catch hallucination, extract through-lines, fix inputs/rules).

Only the Forecaster's `P(YES)` is forward-graded - so "raw model" vs "model+web-research"
vs "pure rule" are competing **input strategies** tested through one harness, decided by
the ledger, not by argument.

## The rubric (every experiment is scored on these)

1. **Power** - no GO/NO-GO on n<~40 or <2 SE (small samples are the most common way a
   backtest lies to you).
2. **Out-of-sample / forward** - confirm on fresh/disjoint/future data, never the data the
   edge was found on (in-sample edges routinely vanish out-of-sample).
3. **Executable** - survives an order-book *walk* at real size after spread (a mid-quote
   "edge" can be pure spread; depth and crossing cost are part of the result).
4. **Multiple-testing discipline** - running many experiments manufactures false positives;
   promotion requires forward confirmation + FDR awareness, not in-sample p.
5. **Edge-type named** - predictive / structural / latency / calibration / resolution-
   artifact (you should know which mechanism you think you are exploiting).
6. **Data is real & point-in-time** - machine-readable, no leakage (no method rescues
   unextractable inputs or future-leaked data).
7. **Observation-level reviewed** - read the actual cases, not just aggregate Brier (how you
   catch a model reasoning confidently over garbage inputs).
8. **Inefficiency × our-advantage** - category fit: both the market's inefficiency and your
   own advantage in it should be positive.

## Non-goals (for now)
No live trading. No infra we don't need yet. No category connector before an experiment
needs it. **Experiment-first** - the failure mode is a beautiful machine that never runs.
