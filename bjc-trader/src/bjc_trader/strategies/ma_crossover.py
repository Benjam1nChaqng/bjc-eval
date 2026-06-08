"""Moving-average crossover: BUY when fast MA crosses above slow MA, SELL on cross below."""

from __future__ import annotations

import pandas as pd

from .base import Signal, SignalType, Strategy, validate_bars


class MACrossover(Strategy):
    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("window lengths must be >= 1")
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, symbol: str, bars: pd.DataFrame) -> list[Signal]:
        validate_bars(bars)
        if len(bars) < self.slow + 1:
            return []

        close = bars["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        # Position of fast relative to slow; a change in sign is a crossover.
        diff = (fast_ma - slow_ma).dropna()
        side = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        signals: list[Signal] = []
        prev = None
        for ts, cur in side.items():
            if prev is not None and cur != prev and cur != 0:
                sig_type = SignalType.BUY if cur > 0 else SignalType.SELL
                signals.append(
                    Signal(
                        symbol=symbol,
                        type=sig_type,
                        timestamp=pd.Timestamp(ts).to_pydatetime(),
                        strength=min(1.0, abs(float(diff.loc[ts])) / float(close.loc[ts]) * 10),
                        reason=f"{self.fast}/{self.slow} MA cross {'up' if cur > 0 else 'down'}",
                        strategy_name=self.name,
                    )
                )
            prev = cur
        return signals
