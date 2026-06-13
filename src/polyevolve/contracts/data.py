from typing import Any, Protocol


class DataSource(Protocol):
    name: str

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]: ...

    def render(self, payload: dict[str, Any]) -> str:
        """Turn a fetch payload into the text the agent reads in its prompt."""
        ...
