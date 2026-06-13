"""Plugin registry - the seam that keeps core ignorant of plugins.

Each plugin class decorates itself with @register_market / @register_connector /
@register_forecaster, which stores it under a string key. `discover()` imports
every module under markets/ connectors/ forecasters/, triggering those decorators
as a side effect - so a fresh `import` of this module followed by `discover()`
auto-loads every plugin on disk. CORE NEVER IMPORTS A SPECIFIC PLUGIN: discovery
walks the package directories generically, and plugins import core, never the
reverse. That one-way dependency is what makes adding a plugin safe.

The decorators return the class unchanged, so they're transparent to the plugin.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import TypeVar

from .interfaces import Forecaster, MarketSource, ResearchConnector

_MARKETS: dict[str, type[MarketSource]] = {}
_CONNECTORS: dict[str, type[ResearchConnector]] = {}
_FORECASTERS: dict[str, type[Forecaster]] = {}

M = TypeVar("M", bound=type[MarketSource])
C = TypeVar("C", bound=type[ResearchConnector])
F = TypeVar("F", bound=type[Forecaster])


def register_market(key: str) -> Callable[[M], M]:
    """Class decorator: register a MarketSource implementation under `key`."""

    def deco(cls: M) -> M:
        if key in _MARKETS:
            raise ValueError(f"duplicate market source key: {key!r}")
        _MARKETS[key] = cls
        return cls

    return deco


def register_connector(key: str) -> Callable[[C], C]:
    """Class decorator: register a ResearchConnector implementation under `key`."""

    def deco(cls: C) -> C:
        if key in _CONNECTORS:
            raise ValueError(f"duplicate connector key: {key!r}")
        _CONNECTORS[key] = cls
        return cls

    return deco


def register_forecaster(key: str) -> Callable[[F], F]:
    """Class decorator: register a Forecaster implementation under `key`."""

    def deco(cls: F) -> F:
        if key in _FORECASTERS:
            raise ValueError(f"duplicate forecaster key: {key!r}")
        _FORECASTERS[key] = cls
        return cls

    return deco


def get_market(key: str) -> type[MarketSource]:
    return _MARKETS[key]


def get_connector(key: str) -> type[ResearchConnector]:
    return _CONNECTORS[key]


def get_forecaster(key: str) -> type[Forecaster]:
    return _FORECASTERS[key]


def all_markets() -> dict[str, type[MarketSource]]:
    return dict(_MARKETS)


def all_connectors() -> dict[str, type[ResearchConnector]]:
    return dict(_CONNECTORS)


def all_forecasters() -> dict[str, type[Forecaster]]:
    return dict(_FORECASTERS)


# Plugin packages discovery walks. Each is a subpackage of `polyevolve`; importing
# every module under them fires the registration decorators as a side effect.
_PLUGIN_PACKAGES = ("markets", "connectors", "forecasters")


def discover() -> None:
    """Import every plugin module so its registration decorator runs.

    Idempotent: re-importing an already-imported module is a no-op, and the
    decorators guard against duplicate keys. Call once after importing the
    registry to populate all_*; safe to call again.
    """
    for pkg_name in _PLUGIN_PACKAGES:
        full = f"polyevolve.{pkg_name}"
        try:
            pkg = importlib.import_module(full)
        except ModuleNotFoundError:
            continue
        if not hasattr(pkg, "__path__"):
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.startswith("_"):
                continue
            importlib.import_module(f"{full}.{mod.name}")
