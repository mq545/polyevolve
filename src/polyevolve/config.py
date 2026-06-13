from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Polite-pool API etiquette (Wikipedia, Wikimedia pageviews, etc.) asks for a contact
# in the User-Agent. Set POLYEVOLVE_CONTACT to your own email or project URL; the default is
# the project repo so the UA is always a reachable contact.
CONTACT = os.environ.get("POLYEVOLVE_CONTACT", "https://github.com/mq545/polyevolve")
USER_AGENT = f"polyevolve-forecaster/0.1 ({CONTACT})"


@dataclass(frozen=True)
class Config:
    db_url: str
    default_model: str
    kalshi_api_key: str | None
    kalshi_private_key_path: Path | None
    newsapi_key: str | None

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()

        kalshi_pk = os.environ.get("KALSHI_PRIVATE_KEY_PATH")

        return cls(
            db_url=os.environ.get(
                "DB_URL",
                "postgresql://superpod:superpod@localhost:5432/superpod",
            ),
            default_model=os.environ.get(
                "DEFAULT_MODEL", "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M"
            ),
            kalshi_api_key=os.environ.get("KALSHI_API_KEY") or None,
            kalshi_private_key_path=Path(kalshi_pk) if kalshi_pk else None,
            newsapi_key=os.environ.get("NEWSAPI_KEY") or None,
        )
