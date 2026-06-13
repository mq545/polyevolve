"""ShinkaEvolve adapter: full-program (EVOLVE-BLOCK) evolution behind the optimizer API.

Importing this package never imports shinka itself - the dependency lives in a separate venv
and is reached only by subprocess (see `adapter.ShinkaEvolveOptimizer`). The program format
and the genome loader (`program`) are pure-polyevolve and safe to import anywhere.
"""

from __future__ import annotations

from polyevolve.evolve.shinka.adapter import ShinkaEvolveOptimizer
from polyevolve.evolve.shinka.program import (
    load_program_genome,
    seed_program,
    validate_program,
)

__all__ = [
    "ShinkaEvolveOptimizer",
    "load_program_genome",
    "seed_program",
    "validate_program",
]
