"""Optional Langfuse integration.

Off by default. Activates only when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
are set AND the langfuse package is installed (`uv add langfuse`). Any failure
degrades silently to Postgres-only tracing - observability must never break the
pipeline.

Deploy options:
- Langfuse Cloud (fastest): create a project at cloud.langfuse.com, set the two
  keys + LANGFUSE_HOST=https://cloud.langfuse.com.
- Self-hosted: run the Langfuse docker stack and point LANGFUSE_HOST at it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_warned = False


def get_langfuse() -> Any | None:
    global _warned
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse()
    except Exception:
        if not _warned:
            logger.warning(
                "Langfuse keys set but client init failed "
                "(install with `uv add langfuse`). Falling back to Postgres-only tracing."
            )
            _warned = True
        return None
