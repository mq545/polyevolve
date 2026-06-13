"""ShinkaEvolve adapter tests that need NO shinka venv and NO model server.

The bridge contract test simulates how ShinkaEvolve invokes the evaluator: it shells out to
`bridge.py` with a model-free constant program and a tiny frozen dataset, and asserts the two
result files come out in the exact shape shinka 0.0.6 reads back (metrics.json with
combined_score = val score, public = train, private = val; correct.json = {correct, error}).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_CONST_PROGRAM = (
    "from polyevolve.reason.dsl import Forecast\n"
    "# EVOLVE-BLOCK-START\n"
    "def forecast(q, pool):\n"
    "    return Forecast(p_yes=0.5, size=0.0, confidence=0.5)\n"
    "# EVOLVE-BLOCK-END\n"
)


def _row(i: int, outcome: bool, price: float) -> dict:
    return {
        "question": {
            "id": f"q{i}",
            "text": "t?",
            "as_of": "2025-01-01T00:00:00Z",
            "outcome": outcome,
            "market_price": price,
        },
        "pool": {"items": []},
    }


def test_bridge_contract_offline(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    prog = tmp_path / "const.py"
    prog.write_text(_CONST_PROGRAM)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "train": [_row(1, True, 0.6), _row(2, False, 0.4)],
                "val": [_row(3, True, 0.55), _row(4, False, 0.45)],
            }
        )
    )
    results = tmp_path / "res"
    env = {
        **os.environ,
        "POLYEVOLVE_PYTHON": sys.executable,
        "POLYEVOLVE_REPO": str(repo),
        "POLYEVOLVE_DATASET": str(dataset),
        "POLYEVOLVE_OBJECTIVE": "calibration",
    }
    bridge = repo / "src" / "polyevolve" / "evolve" / "shinka" / "bridge.py"
    proc = subprocess.run(
        [sys.executable, str(bridge), "--program_path", str(prog), "--results_dir", str(results)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    metrics = json.loads((results / "metrics.json").read_text())
    correct = json.loads((results / "correct.json").read_text())

    # constant 0.5 forecaster -> Brier 0.25 on each split -> combined_score (val) = -0.25
    assert metrics["combined_score"] == -0.25
    assert metrics["public"]["train_brier"] == 0.25  # mutator sees train
    assert metrics["private"]["val_brier"] == 0.25  # holdout hidden from mutator
    assert correct == {"correct": True, "error": None}
