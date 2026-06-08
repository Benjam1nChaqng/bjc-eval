"""Name -> Strategy factory so the CLI can build strategies by string."""

from __future__ import annotations

from collections.abc import Callable

from .base import Strategy
from .ma_crossover import MACrossover
from .rsi_reversion import RSIReversion

_REGISTRY: dict[str, Callable[..., Strategy]] = {
    MACrossover.name: MACrossover,
    RSIReversion.name: RSIReversion,
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(name: str, **params: object) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    return _REGISTRY[name](**params)
