"""Agentic, leakage-safe data gathering - the `research` node.

The rest of the node library reasons over a FIXED `EvidencePool`. This module lets the genome
gather its own evidence: the model is shown the available `as_of`-safe retrieval tools, emits a
structured RETRIEVAL PLAN (which connectors to hit, for which entities), we execute it strictly
``<= question.as_of``, show the model what came back, and let it request another round. The
gathered items become `state.selected` (and are appended to `state.pool`), so the downstream
pipeline (validate -> features -> estimate) runs over agent-gathered data.

LEAKAGE is enforced at the TOOL BOUNDARY, on purpose (not inside a third-party agent framework):
``ToolRegistry.run`` ALWAYS passes ``as_of`` to the connector, every connector returns only
data timestamped on/before it, and every gathered item is stamped with ``as_of``. A live web
search (which would leak the outcome on a historical market) is therefore simply not in the
registry. On forward/live markets the same code is valid because the outcome hasn't happened.

We deliberately roll our own loop on `Model.complete_with_tool` rather than use LangChain/etc:
owning dispatch is what lets us guarantee the `as_of` boundary, it stays a `Node` in the DSL
(evolvable + MockModel-testable), and it adds no heavy deps. The plan->execute->refine shape
(structured plan instead of free-form native tool-calling) is robust to flaky local tool-calling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from polyevolve.models import build_model
from polyevolve.reason.dsl import EvidenceItem, Node, Question, ReasoningState

if TYPE_CHECKING:
    from polyevolve.contracts import DataSource

__all__ = ["RetrievalTool", "ToolRegistry", "default_registry", "research"]


@dataclass
class RetrievalTool:
    """One `as_of`-safe retrieval tool: a named, described wrapper over a `DataSource`.

    ``run`` builds the connector context with the agent-chosen ``entities`` (passed as tags so
    local-language / entity routing works) and the HARD ``as_of`` cutoff, then returns the
    connector's rendered text. The connector is responsible for returning only data <= as_of;
    this wrapper guarantees the cutoff is always supplied.
    """

    name: str
    description: str
    source: DataSource

    def run(self, question: Question, as_of: datetime, entities: Sequence[str]) -> str:
        tags = list(entities) if entities else ([question.category] if question.category else [])
        ctx: dict[str, Any] = {
            "question": question.text,
            "as_of": as_of,  # leakage cutoff - ALWAYS passed
            "tags": tags,
        }
        payload = self.source.fetch(ctx)
        if isinstance(payload, dict) and payload.get("error"):
            return f"[SOURCE ERROR] {self.name}: {payload['error']}"
        return self.source.render(payload)


class ToolRegistry:
    """The set of `as_of`-safe tools the research agent may call. Dispatch enforces the cutoff."""

    def __init__(self, tools: Sequence[RetrievalTool]) -> None:
        self._tools = {t.name: t for t in tools}

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def run(self, name: str, question: Question, as_of: datetime, entities: Sequence[str]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[NO SUCH TOOL] {name}"
        try:
            return tool.run(question, as_of, entities)
        except Exception as exc:  # noqa: BLE001 - one tool failing must not kill the agent
            return f"[SOURCE ERROR] {name}: exception {exc!r}"


def default_registry(*, enable_markets: bool = True) -> ToolRegistry:
    """Build the production registry from the leakage-safe connectors.

    All four are `as_of`-parameterized and point-in-time (Wikipedia revision-before / Yahoo
    bars-before), so they are valid on historical backtests AND forward.
    """
    from polyevolve.data_sources.gdelt_doc import GdeltDocSource
    from polyevolve.data_sources.pageviews import WikipediaPageviewsSource
    from polyevolve.data_sources.polls import WikipediaPollsSource

    tools = [
        RetrievalTool(
            "news",
            "Recent relevant news (GDELT) for given entities/country - events, momentum.",
            GdeltDocSource(),
        ),
        RetrievalTool(
            "polls",
            "Numeric opinion-poll tables (pollster, date, party %) for an election - the "
            "decisive signal for electoral questions. Provide the country/election as entities.",
            WikipediaPollsSource(),
        ),
        RetrievalTool(
            "pageviews",
            "Wikipedia attention/momentum for named entities (people/parties) - a salience proxy.",
            WikipediaPageviewsSource(),
        ),
    ]
    if enable_markets:
        from polyevolve.data_sources.finmarkets import FinancialMarketsSource

        tools.append(
            RetrievalTool(
                "markets",
                "Pre-as_of financial-market moves (equity index + FX + defense/oil/gold basket) "
                "for a country/conflict - a leading macro indicator.",
                FinancialMarketsSource(),
            )
        )
    return ToolRegistry(tools)


_PLAN_TOOL: dict[str, Any] = {
    "name": "submit_research_plan",
    "description": (
        "Choose which retrieval tools to call, for which entities, to best answer the question. "
        "The only output channel."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "description": "A tool name from the list."},
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Specific entities to fetch for - e.g. country, election name, "
                                "party/candidate names. Drives routing/language."
                            ),
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["tool", "entities"],
                },
                "description": "1-4 retrieval requests, most decisive first.",
            },
            "enough": {
                "type": "boolean",
                "description": "True if the evidence gathered so far is already sufficient.",
            },
        },
        "required": ["requests"],
    },
}


def research(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
    registry: ToolRegistry | None = None,
    max_rounds: int = 2,
    max_requests_per_round: int = 4,
) -> Node:
    """Agentic, leakage-safe evidence gathering (plan -> execute <= as_of -> refine).

    The model plans which `as_of`-safe tools to call for which entities; we execute strictly
    within the cutoff and feed results back; it may request another round or stop (``enough``).
    Gathered items replace `state.selected` and are merged into `state.pool` (one item per
    tool, latest wins), so the downstream pipeline reasons over agent-gathered data. Writes
    beliefs['research_log'].
    """
    sys_prompt = (
        "You are a forecasting research lead. You can call point-in-time data tools. Plan the "
        "MINIMAL set of tool calls that would most reduce uncertainty about this question, "
        "choosing precise entities (country, election, parties/candidates). For electoral "
        "questions, polls are usually decisive. Do not pad. Call submit_research_plan once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        reg = registry if registry is not None else default_registry()
        model = build_model(model_id=model_id, anthropic_api_key=anthropic_api_key)
        as_of = state.question.as_of
        gathered: dict[str, EvidenceItem] = {}
        log: list[str] = []

        for rnd in range(1, max_rounds + 1):
            have = (
                "\n".join(f"[{k}] {v.text[:200]}" for k, v in gathered.items())
                if gathered
                else "(nothing yet)"
            )
            user = (
                f"QUESTION: {state.question.text}\n"
                f"AS OF (cutoff): {as_of.date().isoformat()}\n\n"
                f"AVAILABLE TOOLS:\n{reg.describe()}\n\n"
                f"EVIDENCE GATHERED SO FAR:\n{have}\n\n"
                f"Round {rnd}/{max_rounds}. Plan the next retrievals (or set enough=true)."
            )
            try:
                res = model.complete_with_tool(
                    cached_system_blocks=[sys_prompt],
                    user_content=user,
                    tool=_PLAN_TOOL,
                    metadata={"question_id": state.question.id, "node": f"research.r{rnd}"},
                )
                out = res["input"]
            except Exception as exc:  # noqa: BLE001 - planner failed: stop, use what we have
                log.append(f"r{rnd} plan error {exc!r}")
                break
            reqs = [r for r in out.get("requests", []) if isinstance(r, dict)][
                :max_requests_per_round
            ]
            for r in reqs:
                name = str(r.get("tool", "")).strip()
                if name not in reg.names():
                    log.append(f"r{rnd} skip unknown tool {name!r}")
                    continue
                ents = [str(e) for e in r.get("entities", []) if str(e).strip()]
                text = reg.run(name, state.question, as_of, ents).strip()
                if text and "[SOURCE ERROR]" not in text and "[NO SUCH TOOL]" not in text:
                    gathered[name] = EvidenceItem(text=text, source=name, date=as_of)
                    log.append(f"r{rnd} {name}({','.join(ents) or 'default'}) -> {len(text)}c")
                else:
                    log.append(f"r{rnd} {name} -> empty/err")
            if out.get("enough") or not reqs:
                break

        items = list(gathered.values())
        # merge into pool (replace same-source items) + set selected to the agent's gather.
        kept_pool = [it for it in state.pool.items if it.source not in gathered]
        state.pool.items = kept_pool + items
        state.selected = items
        state.beliefs["research_log"] = "; ".join(log)
        return state.log(
            f"research: gathered {len(items)} sources over {min(rnd, max_rounds)} round(s)"
        )

    return _node
