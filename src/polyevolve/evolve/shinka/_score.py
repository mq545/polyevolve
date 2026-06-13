"""Score one evolved program against a frozen dataset - runs in the POLYEVOLVE venv.

The ShinkaEvolve bridge (``bridge.py``, running in the shinka venv) shells out to this
module:

    python -m polyevolve.evolve.shinka._score <program_path> <dataset_path> <objective>

It loads the program as a genome, scores it on the frozen train and val splits through
the same `evaluate_genome` the in-process optimizer uses, and prints ONE json line:

    POLYEVOLVE_RESULT {"combined_score":.., "public":{..train..}, "private":{..val..}, ..}

``combined_score`` is the val (holdout) score - the signal ShinkaEvolve selects on - so the
mutator (which sees only ``public``) cannot overfit to the held-out set.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from polyevolve.evolve.shinka.program import load_program_genome
from polyevolve.reason.dsl import EvidencePool, Question


def _load_split(rows: list[dict[str, Any]]) -> tuple[list[Question], list[EvidencePool]]:
    qs = [Question.model_validate(r["question"]) for r in rows]
    pools = [EvidencePool.model_validate(r["pool"]) for r in rows]
    return qs, pools


def main(argv: list[str]) -> int:
    program_path, dataset_path, objective = argv[1], argv[2], argv[3]

    from polyevolve.evolve.scoring import evaluate_genome  # local: keep import cost off failures

    with open(dataset_path) as fh:
        data = json.load(fh)
    train_qs, train_pools = _load_split(data["train"])
    val_qs, val_pools = _load_split(data["val"])

    genome = load_program_genome(program_path)
    train_m = evaluate_genome(genome, train_qs, train_pools, objective=objective)
    val_m = evaluate_genome(genome, val_qs, val_pools, objective=objective)

    out = {
        "combined_score": float(val_m["combined_score"]),
        "public": {f"train_{k}": v for k, v in train_m.items()},
        "private": {f"val_{k}": v for k, v in val_m.items()},
        "n_train": len(train_qs),
        "n_val": len(val_qs),
    }
    print("POLYEVOLVE_RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
