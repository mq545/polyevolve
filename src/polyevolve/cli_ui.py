"""Tiny, dependency-free terminal-UI helpers for the polyevolve CLI.

Rounded box panels, a unicode sparkline, and ANSI color - styled like Claude Code's
framed output. Color is auto-disabled when stdout is not a TTY (so piped / CI / captured
output stays clean), and all width math is ANSI-aware so colored cells still align.
"""

from __future__ import annotations

import re
import sys

_TTY = sys.stdout.isatty()
_CODES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

_BARS = "▁▂▃▄▅▆▇█"


def color(text: str, name: str) -> str:
    """Wrap ``text`` in an ANSI color (no-op when not a TTY or color unknown)."""
    if not _TTY or name not in _CODES:
        return text
    return f"{_CODES[name]}{text}{_CODES['reset']}"


def _vlen(s: str) -> int:
    """Visible length, ignoring ANSI escapes."""
    return len(_ANSI_RE.sub("", s))


def panel(rows: list[str], *, title: str = "") -> str:
    """Render ``rows`` inside a rounded box, with an optional left-aligned ``title``."""
    inner = max([_vlen(r) for r in rows] + [_vlen(title) + 4, 20])
    if title:
        head = f"╭─ {color(title, 'bold')} "
        top = head + "─" * (inner + 2 - _vlen(head) + 1) + "╮"
    else:
        top = "╭" + "─" * (inner + 2) + "╮"
    body = ["│ " + r + " " * (inner - _vlen(r)) + " │" for r in rows]
    bottom = "╰" + "─" * (inner + 2) + "╯"
    return "\n".join([top, *body, bottom])


def table(headers: list[str], rows: list[list[str]], *, title: str = "") -> str:
    """Render a boxed table with right-aligned numeric-looking cells, ANSI-aware widths.

    A cell is right-aligned when its visible text (ignoring color) parses as a number;
    text cells stay left-aligned. The header row is bolded and rule-separated.
    """

    def _isnum(s: str) -> bool:
        try:
            float(_ANSI_RE.sub("", s).replace("+", "").replace("%", ""))
            return True
        except ValueError:
            return False

    ncol = len(headers)
    widths = [_vlen(headers[i]) for i in range(ncol)]
    for r in rows:
        for i in range(ncol):
            widths[i] = max(widths[i], _vlen(r[i]) if i < len(r) else 0)

    def _cell(text: str, i: int, *, head: bool = False) -> str:
        pad = widths[i] - _vlen(text)
        body = (" " * pad + text) if (not head and _isnum(text)) else (text + " " * pad)
        return color(body, "bold") if head else body

    head = "  ".join(_cell(h, i, head=True) for i, h in enumerate(headers))
    body = ["  ".join(_cell(r[i] if i < len(r) else "", i) for i in range(ncol)) for r in rows]
    return panel([head, "─" * _vlen(head), *body], title=title)


def spark(values: list[float]) -> str:
    """A unicode sparkline for a small numeric series (min->max scaled)."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _BARS[0] * len(values)
    return "".join(_BARS[round((v - lo) / (hi - lo) * (len(_BARS) - 1))] for v in values)
