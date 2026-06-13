"""The EVOLUTION loop - mutate the seed genome's knobs, select on train, report val.

Public surface:
  - fitness / make_calibration_fitness / FitnessFn : the (pluggable) scalar to maximize
  - Optimizer / Result / run_evolution             : the built-in evolutionary search
  - EvolutionOptimizer                             : OO wrapper over run_evolution
  - ShinkaEvolveOptimizer                          : full-program search (Sakana ShinkaEvolve)

Calibration-first today (`-brier`); a return-based fitness swaps in behind `FitnessFn`.
"""

from __future__ import annotations

from .fitness import (
    WORST_FITNESS,
    FitnessFn,
    fitness,
    make_calibration_fitness,
    make_return_fitness,
)
from .optimizer import (
    EvolutionOptimizer,
    Individual,
    Optimizer,
    Result,
    knob_complexity,
    run_evolution,
)
from .shinka import ShinkaEvolveOptimizer

__all__ = [
    "fitness",
    "make_calibration_fitness",
    "make_return_fitness",
    "FitnessFn",
    "WORST_FITNESS",
    "run_evolution",
    "Optimizer",
    "Result",
    "Individual",
    "EvolutionOptimizer",
    "ShinkaEvolveOptimizer",
    "knob_complexity",
]
