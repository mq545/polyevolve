"""The REASONING layer - the genome contract, node library, and seed genome.

A *genome* is an evolvable ``forecast(Question, EvidencePool) -> Forecast`` function
(see :mod:`polyevolve.reason.dsl`) composed from a frozen, unit-tested vocabulary of typed
``Node`` primitives (:mod:`polyevolve.reason.nodes`). The seed (:mod:`polyevolve.reason.seed`)
is one such composition with an EVOLVE-BLOCK body and tunable ``SeedKnobs`` that the
:mod:`polyevolve.evolve` optimizer mutates; :mod:`polyevolve.bench` scores it.

Public surface:
  - dsl  : EvidenceItem / EvidencePool / Question / Forecast / ReasoningState / Node / Genome
  - nodes: the node factories (call_model, ensemble, select_evidence, calibrate, ...)
  - seed : SeedKnobs / forecast / make_seed_genome / run_genome
"""

from __future__ import annotations

from .dsl import (
    EvidenceItem,
    EvidencePool,
    Forecast,
    Genome,
    Node,
    Question,
    ReasoningState,
)
from .nodes import (
    abstain,
    calibrate,
    call_model,
    debate_critique,
    decompose,
    ensemble,
    latent_to_prob,
    select_evidence,
    size_by_edge,
)
from .seed import SeedKnobs, forecast, make_seed_genome, run_genome

__all__ = [
    # dsl (frozen contract)
    "EvidenceItem",
    "EvidencePool",
    "Forecast",
    "Genome",
    "Node",
    "Question",
    "ReasoningState",
    # nodes (frozen primitives)
    "abstain",
    "calibrate",
    "call_model",
    "debate_critique",
    "decompose",
    "ensemble",
    "latent_to_prob",
    "select_evidence",
    "size_by_edge",
    # seed genome
    "SeedKnobs",
    "forecast",
    "make_seed_genome",
    "run_genome",
]
