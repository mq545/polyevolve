"""Evolution layer: genome, evaluator, and the fast-iteration harness.

The Genome is the evolvable artifact (prompt + config). The evaluator scores a
(genome, model) candidate against a frozen eval snapshot, caching predictions so
re-evaluating an unchanged candidate is free. This is the substrate ShinkaEvolve
plugs into - see reference_shinkaevolve in memory.
"""

from .evaluator import EvalResult, evaluate
from .genome import Genome

__all__ = ["EvalResult", "Genome", "evaluate"]
