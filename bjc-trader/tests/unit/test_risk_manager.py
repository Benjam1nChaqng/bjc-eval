from __future__ import annotations

from datetime import datetime

from bjc_trader.execution.broker import Account, OrderRequest, OrderSide, Position
from bjc_trader.risk.manager import RiskManager
from bjc_trader.strategies.base import Signal, SignalType


def _account(equity=10_000.0, cash=10_000.0, blocked=False):
    return Account(equity=equity, cash=cash, buying_power=cash, blocked=blocked)


def _buy(symbol="AAA", qty=1, price=100.0):
    return OrderRequest(symbol, OrderSide.BUY, qty, price)


def _sell(symbol="AAA", qty=1, price=100.0):
    return OrderRequest(symbol, OrderSide.SELL, qty, price)


def test_valid_buy_approved(settings):
    rm = RiskManager(settings)
    assert rm.check(_buy(qty=5, price=100), _account(), []).approved


def test_no_shorting(settings):
    rm = RiskManager(settings)
    # Sell 5 when holding 2 -> would short.
    d = rm.check(_sell(qty=5), _account(), [Position("AAA", 2, 100.0)])
    assert not d.approved
    assert any("short" in r for r in d.reasons)


def test_sell_within_holding_allowed(settings):
    rm = RiskManager(settings)
    assert rm.check(_sell(qty=2), _account(), [Position("AAA", 5, 100.0)]).approved


def test_no_margin(settings):
    rm = RiskManager(settings)
    d = rm.check(_buy(qty=10, price=100), _account(cash=500.0, equity=10_000.0), [])
    assert not d.approved
    assert any("insufficient cash" in r for r in d.reasons)


def test_per_trade_notional_cap(settings):
    settings.max_trade_notional = 100.0
    rm = RiskManager(settings)
    d = rm.check(_buy(qty=5, price=100), _account(), [])  # notional 500 > 100
    assert not d.approved
    assert any("per-trade cap" in r for r in d.reasons)


def test_position_pct_cap(settings):
    settings.max_position_pct = 10.0  # 10% of 10k = 1000 max position value
    settings.max_trade_notional = 100_000.0
    rm = RiskManager(settings)
    d = rm.check(_buy(qty=20, price=100), _account(cash=10_000.0), [])  # 2000 > 1000
    assert not d.approved
    assert any("equity cap" in r for r in d.reasons)


def test_max_open_positions(settings):
    settings.max_open_positions = 1
    rm = RiskManager(settings)
    held = [Position("BBB", 1, 100.0)]
    d = rm.check(_buy(symbol="AAA", qty=1, price=100), _account(), held)
    assert not d.approved
    assert any("max open positions" in r for r in d.reasons)


def test_daily_loss_engages_kill_switch(settings):
    settings.max_daily_loss = 100.0
    rm = RiskManager(settings)
    d = rm.check(_buy(qty=1, price=100), _account(), [], daily_pnl=-150.0)
    assert not d.approved
    assert rm.kill_switch_engaged
    # Kill switch latches: a subsequent fine order is still rejected.
    assert not rm.check(_buy(qty=1, price=100), _account(), [], daily_pnl=0.0).approved


def test_blocked_account_rejected(settings):
    rm = RiskManager(settings)
    assert not rm.check(_buy(), _account(blocked=True), []).approved


def test_sizing_respects_caps(settings):
    settings.max_trade_notional = 250.0
    rm = RiskManager(settings)
    sig = Signal("AAA", SignalType.BUY, datetime.utcnow(), 1.0, "x", "s")
    order = rm.size_order(sig, _account(equity=10_000.0), price=100.0)
    assert order.qty == 2  # 250 // 100
    assert order.notional <= 250.0
