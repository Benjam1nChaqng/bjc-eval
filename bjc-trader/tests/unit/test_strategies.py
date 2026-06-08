from __future__ import annotations

import numpy as np
import pytest

from bjc_trader.strategies import build
from bjc_trader.strategies.base import SignalType
from bjc_trader.strategies.ma_crossover import MACrossover
from bjc_trader.strategies.rsi_reversion import RSIReversion, rsi
from tests.conftest import make_bars


def test_ma_crossover_emits_buy_on_upcross():
    bars = make_bars([10] * 30 + list(np.linspace(10, 30, 40)))
    sigs = MACrossover(fast=5, slow=20).generate_signals("AAA", bars)
    assert sigs
    assert sigs[0].type is SignalType.BUY
    # Strategies never emit short intent — only BUY/SELL/HOLD.
    assert all(s.type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD) for s in sigs)


def test_ma_crossover_emits_sell_on_downcross():
    # Rise (fast crosses above) then fall (fast crosses back below) -> a SELL must fire.
    bars = make_bars(list(np.linspace(10, 30, 40)) + list(np.linspace(30, 10, 40)))
    sigs = MACrossover(fast=5, slow=20).generate_signals("AAA", bars)
    assert any(s.type is SignalType.SELL for s in sigs)


def test_ma_crossover_too_few_bars_returns_empty():
    bars = make_bars([10, 11, 12])
    assert MACrossover(fast=5, slow=20).generate_signals("AAA", bars) == []


def test_ma_crossover_rejects_bad_windows():
    with pytest.raises(ValueError):
        MACrossover(fast=50, slow=20)


def test_rsi_bounds():
    bars = make_bars(list(np.linspace(10, 40, 60)))
    values = rsi(bars["close"], 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_rsi_reversion_buys_after_selloff_recovery():
    # Sharp drop (oversold) then recovery should fire a BUY when RSI climbs back through 30.
    closes = list(np.linspace(50, 20, 20)) + list(np.linspace(20, 45, 20))
    strat = RSIReversion(period=14, oversold=30, overbought=70)
    sigs = strat.generate_signals("AAA", make_bars(closes))
    assert any(s.type is SignalType.BUY for s in sigs)


def test_registry_builds_known_strategy():
    strat = build("ma_crossover", fast=10, slow=30)
    assert strat.name == "ma_crossover"


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        build("does_not_exist")
