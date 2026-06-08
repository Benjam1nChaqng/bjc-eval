"""Strategy interface and the Signal type.

Strategies are *pure*: they take a price DataFrame and return intent (BUY/SELL/HOLD).
They never size positions, touch a broker, or emit short signals. This purity is what
makes them trivially testable and safe to backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    symbol: str
    type: SignalType
    timestamp: datetime
    strength: float  # 0.0–1.0 confidence; informational only
    reason: str
    strategy_name: str


@runtime_checkable
class Strategy(Protocol):
    name: str

    def generate_signals(self, symbol: str, bars: pd.DataFrame) -> list[Signal]:
        """Return signals for ``symbol`` given OHLCV ``bars`` indexed by timestamp."""
        ...


def validate_bars(bars: pd.DataFrame) -> None:
    """Raise if the frame is missing required OHLCV columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
