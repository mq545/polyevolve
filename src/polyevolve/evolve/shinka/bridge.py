"""ShinkaEvolve eval entrypoint - the venv-boundary bridge. STDLIB ONLY.

ShinkaEvolve runs in its own venv (it pins httpx 0.27, conflicting with our litellm 0.28)
and invokes this file as a subprocess: ``python bridge.py --program_path X --results_dir Y``.
This file MUST NOT import polyevolve or shinka - it only shells across to the polyevolve venv
to do the real scoring, then writes the two result files ShinkaEvolve reads back.

Config via env (set by the adapter / launcher):
  POLYEVOLVE_PYTHON    path to the polyevolve venv python
  POLYEVOLVE_REPO      repo root (PYTHONPATH=<repo>/src for the scorer)
  POLYEVOLVE_DATASET   path to the frozen dataset.json (train/val splits)
  POLYEVOLVE_OBJECTIVE "calibration" | "return"

Result-file contract (verified against shinka 0.0.6 wrap_eval.py / async_runner.py):
  metrics.json  = {"combined_score": float, "public": {...}, "private": {...}}
  correct.json  = {"correct": bool, "error": str|None}     <- MUST be this dict shape
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ERROR: dict[str, Any] = {"combined_score": -1.0, "public": {"error": True}, "private": {}}


def _run(program_path: str) -> dict[str, Any]:
    py = os.environ["POLYEVOLVE_PYTHON"]
    repo = os.environ["POLYEVOLVE_REPO"]
    dataset = os.environ["POLYEVOLVE_DATASET"]
    objective = os.environ.get("POLYEVOLVE_OBJECTIVE", "calibration")

    env = {**os.environ, "PYTHONPATH": f"{repo}/src"}
    proc = subprocess.run(  # noqa: S603 - args are operator-controlled
        [py, "-m", "polyevolve.evolve.shinka._score", program_path, dataset, objective],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("POLYEVOLVE_EVAL_TIMEOUT", "7200")),
    )
    if proc.returncode != 0:
        return {**_ERROR, "public": {"error": True, "stderr": proc.stderr[-1500:]}}
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("POLYEVOLVE_RESULT ")), None
    )
    if line is None:
        tail = proc.stdout[-800:]
        return {**_ERROR, "public": {"error": True, "msg": "no result line", "out": tail}}
    r = json.loads(line[len("POLYEVOLVE_RESULT ") :])
    return {
        "combined_score": r["combined_score"],
        "public": {**r["public"], "n_train": r.get("n_train", 0)},
        "private": {**r["private"], "n_val": r.get("n_val", 0)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    try:
        metrics = _run(args.program_path)
    except Exception as exc:  # noqa: BLE001 - never crash the eval subprocess
        metrics = {**_ERROR, "public": {"error": True, "exc": repr(exc)}}

    (Path(args.results_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))

    err = metrics.get("public", {}).get("error", False)
    n_val = metrics.get("private", {}).get("n_val", 0)
    correct = bool(not err and n_val > 0)
    error_msg = None if correct else str(metrics.get("public", {}) or "no scored markets")
    (Path(args.results_dir) / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error_msg})
    )
    print(json.dumps({**metrics, "correct": correct}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
