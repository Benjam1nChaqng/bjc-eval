from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bjc_trader.backtest.engine import run_backtest
from bjc_trader.backtest.metrics import max_drawdown, total_return, win_rate
from bjc_trader.strategies.ma_crossover import MACrossover
from tests.conftest import make_bars


def test_total_return_basic():
    eq = pd.Series([100.0, 110.0])
    assert total_return(eq) == pytest.approx(10.0)  # +10%


def test_max_drawdown_basic():
    eq = pd.Series([100.0, 120.0, 60.0, 90.0])
    # Peak 120 -> trough 60 = -50%
    assert round(max_drawdown(eq), 2) == -50.0


def test_win_rate():
    assert win_rate([0.1, -0.2, 0.3]) == pytest.approx(100 * 2 / 3)


def test_backtest_is_deterministic_and_profits_in_uptrend():
    bars = make_bars([10] * 30 + list(np.linspace(10, 30, 40)))
    strat = MACrossover(fast=5, slow=20)
    r1 = run_backtest(strat, "AAA", bars, cash=1000.0)
    r2 = run_backtest(strat, "AAA", bars, cash=1000.0)
    assert r1.final_equity == r2.final_equity  # deterministic
    assert r1.final_equity > 1000.0  # rode the uptrend


def test_commission_reduces_return():
    bars = make_bars([10] * 30 + list(np.linspace(10, 30, 40)))
    strat = MACrossover(fast=5, slow=20)
    no_fee = run_backtest(strat, "AAA", bars, cash=1000.0, commission=0.0)
    with_fee = run_backtest(strat, "AAA", bars, cash=1000.0, commission=0.01)
    assert with_fee.final_equity < no_fee.final_equity
