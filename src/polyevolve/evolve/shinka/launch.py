"""Launch a ShinkaEvolve run - RUNS IN THE SHINKA VENV (imports shinka).

Invoked by the adapter as ``<shinka_python> -m polyevolve.evolve.shinka.launch <config.json>``
(the shinka venv has polyevolve on PYTHONPATH for this module only - it imports shinka, not
the polyevolve runtime). It reads a json config the adapter wrote, builds the runner, and
points the evaluator at ``bridge.py`` (which shells back to the polyevolve venv).

This file is deliberately thin and import-guards shinka so importing the package never
requires the dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    cfg = json.loads(Path(argv[1]).read_text())

    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    bridge = str(Path(__file__).parent / "bridge.py")

    evo = EvolutionConfig(
        task_sys_msg=cfg["task_sys_msg"],
        num_generations=cfg["num_generations"],
        llm_models=cfg["llm_models"],
        init_program_path=cfg["init_program_path"],
        results_dir=cfg["results_dir"],
        embedding_model=cfg.get("embedding_model"),  # None for local Ollama (no embeddings)
        max_novelty_attempts=cfg.get("max_novelty_attempts", 1),
        patch_types=cfg.get("patch_types", ["diff", "full"]),
        patch_type_probs=cfg.get("patch_type_probs", [0.7, 0.3]),
        max_api_costs=cfg.get("max_api_costs"),
    )
    db = DatabaseConfig(
        num_islands=cfg.get("num_islands", 2),
        archive_size=cfg.get("archive_size", 20),
    )
    job = LocalJobConfig(eval_program_path=bridge)

    runner = ShinkaEvolveRunner(
        evo_config=evo,
        job_config=job,
        db_config=db,
        max_evaluation_jobs=cfg.get("max_eval_jobs", 1),
    )
    runner.run()
    print(f"SHINKA_DONE {cfg['results_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
