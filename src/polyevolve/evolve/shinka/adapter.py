"""ShinkaEvolveOptimizer - full-program evolution behind the built-in optimizer's interface.

This adapter lets ShinkaEvolve (Sakana's LLM program-evolution engine) rewrite the entire
`forecast()` pipeline, while the rest of the platform sees only the same
``optimize(seed_knobs, train, val) -> Result`` contract as the built-in `run_evolution`.

ShinkaEvolve runs OUT OF PROCESS in its own venv (it pins httpx 0.27, conflicting with our
litellm). The seam:

    adapter (polyevolve venv)
      -> writes initial.py (seed program), dataset.json (frozen train/val), config.json
      -> subprocess: <shinka_python> launch.py config.json        [shinka venv]
           ShinkaEvolveRunner mutates the EVOLVE-BLOCK, evaluates each child via
           bridge.py  ->  subprocess: <polyevolve_python> -m ..._score   [polyevolve venv]
      -> reads results_dir/best/main.py, re-scores seed + champion in-process -> Result

Nothing in the platform imports shinka; this adapter only shells out to it. Constructing the
adapter validates the shinka python exists, so failures are clear and early.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from polyevolve.evolve.optimizer import Result
from polyevolve.evolve.scoring import Objective, evaluate_genome
from polyevolve.evolve.shinka.program import load_program_genome, seed_program
from polyevolve.reason.dsl import EvidencePool, Question
from polyevolve.reason.seed import SeedKnobs

_DEFAULT_SHINKA_PYTHON = "~/.venvs/shinka/bin/python"
_REPO_ROOT = Path(__file__).resolve().parents[4]

_DEFAULT_TASK_MSG = (
    "You are improving a prediction-market forecasting genome: the body of forecast(), a "
    "pipeline composed from typed nodes, to MAXIMIZE combined_score (calibration = -Brier; "
    "return = net-of-spread ROI) on real resolved markets. Only the EVOLVE-BLOCK region is "
    "editable. The node toolbox and rules are documented in the program's module docstring - "
    "compose those nodes; do not add imports, I/O, or new helpers.\n"
    "WHAT MOVES THE SCORE (do not just rephrase the system prompt):\n"
    "1. Pipeline STRUCTURE - try decompose, ensembling, debate_critique, a latent-margin "
    "estimate (coherent on threshold markets), or validating/feature-extracting evidence "
    "before the estimate. Order matters: an estimate node must precede calibrate/abstain/size.\n"
    "2. select_evidence k and the calibrate coeff (counter overconfidence), and the abstain "
    "gates (only trade when confident AND diverging from the market price).\n"
    "3. reweight_polls to drop captured pollsters on political races.\n"
    "Always end with size_by_edge so the forecast carries a stake. Keep edits minimal and "
    "test one idea at a time."
)


class ShinkaEvolveOptimizer:
    """Full-program optimizer: delegates EVOLVE-BLOCK rewriting to ShinkaEvolve.

    Same interface as the built-in loop (`optimize` returning a `Result`), so
    ``polyevolve evolve --optimizer shinka`` swaps it in with no other code change. The
    champion's evolved source rides on ``Result.champion_source`` / ``champion_path``.
    """

    def __init__(
        self,
        *,
        objective: Objective = "calibration",
        generations: int = 20,
        mutator: str = "local/qwen3:30b-a3b-instruct-2507-q4_K_M@http://localhost:11434/v1",
        num_islands: int = 2,
        archive_size: int = 20,
        max_eval_jobs: int = 1,
        max_api_costs: float | None = None,
        patch_types: list[str] | None = None,
        patch_type_probs: list[float] | None = None,
        shinka_python: str = _DEFAULT_SHINKA_PYTHON,
        work_dir: str | Path | None = None,
        task_sys_msg: str = _DEFAULT_TASK_MSG,
        keep_work_dir: bool = False,
    ) -> None:
        self.objective = objective
        self.generations = generations
        self.mutator = mutator
        self.num_islands = num_islands
        self.archive_size = archive_size
        self.max_eval_jobs = max_eval_jobs
        self.max_api_costs = max_api_costs
        # bias toward FULL rewrites: diffs are brittle on indentation-sensitive Python.
        self.patch_types = patch_types or ["full", "diff"]
        self.patch_type_probs = patch_type_probs or [0.7, 0.3]
        self.task_sys_msg = task_sys_msg
        self.keep_work_dir = keep_work_dir
        self._work_dir = work_dir
        self.shinka_python = Path(shinka_python).expanduser()
        if not self.shinka_python.exists():
            raise FileNotFoundError(
                f"shinka python not found at {self.shinka_python}. ShinkaEvolve needs its own "
                "venv (it pins httpx 0.27, conflicting with our litellm). Create it once:\n"
                "  python -m venv ~/.venvs/shinka\n"
                "  ~/.venvs/shinka/bin/pip install shinka-evolve\n"
                "or pass shinka_python=<path> if it lives elsewhere."
            )

    # -- serialization -----------------------------------------------------------------
    @staticmethod
    def _dump_split(
        qs: Sequence[Question], pools: Sequence[EvidencePool] | None
    ) -> list[dict[str, Any]]:
        pl = list(pools) if pools is not None else [EvidencePool(items=[]) for _ in qs]
        if len(pl) != len(qs):
            raise ValueError("pools must align 1:1 with questions")
        return [
            {"question": q.model_dump(mode="json"), "pool": p.model_dump(mode="json")}
            for q, p in zip(qs, pl, strict=True)
        ]

    def _score_program(
        self,
        path: str | Path,
        train_qs: Sequence[Question],
        train_pools: Sequence[EvidencePool] | None,
        val_qs: Sequence[Question],
        val_pools: Sequence[EvidencePool] | None,
    ) -> tuple[float, float]:
        genome = load_program_genome(path)
        tr = evaluate_genome(genome, list(train_qs), train_pools, objective=self.objective)
        va = evaluate_genome(genome, list(val_qs), val_pools, objective=self.objective)
        return float(tr["combined_score"]), float(va["combined_score"])

    # -- the optimizer interface -------------------------------------------------------
    def optimize(
        self,
        seed_knobs: SeedKnobs,
        train_qs: Sequence[Question],
        val_qs: Sequence[Question],
        *,
        train_pools: Sequence[EvidencePool] | None = None,
        val_pools: Sequence[EvidencePool] | None = None,
    ) -> Result:
        base = Path(self._work_dir) if self._work_dir else Path(tempfile.mkdtemp(prefix="shinka_"))
        task = base / "task"
        task.mkdir(parents=True, exist_ok=True)
        results_dir = base / "results"

        initial = task / "initial.py"
        initial.write_text(seed_program(seed_knobs))
        dataset = task / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "train": self._dump_split(train_qs, train_pools),
                    "val": self._dump_split(val_qs, val_pools),
                }
            )
        )
        config = task / "config.json"
        config.write_text(
            json.dumps(
                {
                    "task_sys_msg": self.task_sys_msg,
                    "num_generations": self.generations,
                    "llm_models": [self.mutator],
                    "init_program_path": str(initial),
                    "results_dir": str(results_dir),
                    "embedding_model": None,
                    "num_islands": self.num_islands,
                    "archive_size": self.archive_size,
                    "max_eval_jobs": self.max_eval_jobs,
                    "max_api_costs": self.max_api_costs,
                    "patch_types": self.patch_types,
                    "patch_type_probs": self.patch_type_probs,
                }
            )
        )

        env = {
            **_os_environ(),
            "POLYEVOLVE_PYTHON": sys.executable,
            "POLYEVOLVE_REPO": str(_REPO_ROOT),
            "POLYEVOLVE_DATASET": str(dataset),
            "POLYEVOLVE_OBJECTIVE": self.objective,
        }
        # local OpenAI-compatible mutators (Ollama/vLLM) need this sentinel key set.
        if self.mutator.startswith("local/"):
            env.setdefault("LOCAL_OPENAI_API_KEY", "local")
        launch = Path(__file__).parent / "launch.py"
        proc = subprocess.run(  # noqa: S603 - operator-controlled args/paths
            [str(self.shinka_python), str(launch), str(config)],
            cwd=str(_REPO_ROOT),
            env=env,
        )
        if proc.returncode != 0:
            log = results_dir / "evolution_run.log"
            raise RuntimeError(f"ShinkaEvolve run failed (exit {proc.returncode}); see {log}")

        champion = results_dir / "best" / "main.py"
        if not champion.exists():
            raise RuntimeError(f"no champion produced at {champion}")

        seed_tr, seed_va = self._score_program(initial, train_qs, train_pools, val_qs, val_pools)
        # If no offspring beat the seed, ShinkaEvolve's champion IS the seed program. With a
        # stochastic forecaster, re-scoring it separately would manufacture a fake delta - so
        # reuse the seed's scores rather than re-running the identical program.
        if champion.read_text() == initial.read_text():
            champ_tr, champ_va = seed_tr, seed_va
        else:
            champ_tr, champ_va = self._score_program(
                champion, train_qs, train_pools, val_qs, val_pools
            )
        return Result(
            best_knobs=seed_knobs,
            best_train_fitness=champ_tr,
            best_val_fitness=champ_va,
            seed_train_fitness=seed_tr,
            seed_val_fitness=seed_va,
            champion_source=champion.read_text(),
            champion_path=str(champion),
        )


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)
