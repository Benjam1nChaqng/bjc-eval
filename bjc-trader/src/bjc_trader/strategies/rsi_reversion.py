"""RSI mean-reversion: BUY when RSI exits oversold, SELL when it exits overbought."""

from __future__ import annotations

import pandas as pd

from .base import Signal, SignalType, Strategy, validate_bars


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


class RSIReversion(Strategy):
    name = "rsi_reversion"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70) -> None:
        if not (0 < oversold < overbought < 100):
            raise ValueError("require 0 < oversold < overbought < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, symbol: str, bars: pd.DataFrame) -> list[Signal]:
        validate_bars(bars)
        values = rsi(bars["close"], self.period).dropna()
        if values.empty:
            return []

        signals: list[Signal] = []
        prev = None
        for ts, cur in values.items():
            if prev is not None:
                # Crossing back up through oversold -> BUY; down through overbought -> SELL.
                if prev <= self.oversold < cur:
                    signals.append(
                        self._sig(symbol, SignalType.BUY, ts, cur, "RSI exit oversold")
                    )
                elif prev >= self.overbought > cur:
                    signals.append(
                        self._sig(symbol, SignalType.SELL, ts, cur, "RSI exit overbought")
                    )
            prev = cur
        return signals

    def _sig(
        self, symbol: str, t: SignalType, ts: object, rsi_val: float, reason: str
    ) -> Signal:
        # Strength scales with distance of RSI from the neutral 50 line.
        strength = min(1.0, abs(float(rsi_val) - 50) / 50)
        return Signal(
            symbol=symbol,
            type=t,
            timestamp=pd.Timestamp(ts).to_pydatetime(),
            strength=strength,
            reason=f"{reason} (RSI={float(rsi_val):.1f})",
            strategy_name=self.name,
        )
