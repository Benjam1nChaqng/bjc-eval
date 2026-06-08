from __future__ import annotations

from datetime import datetime

from bjc_trader.execution.engine import ExecutionEngine
from bjc_trader.risk.manager import RiskManager
from bjc_trader.strategies.base import Signal, SignalType


def _signal(symbol="AAA", t=SignalType.BUY):
    return Signal(symbol, t, datetime.utcnow(), 1.0, "reason", "ma_crossover")


def test_paper_broker_submits_immediately(settings, paper_broker, store):
    engine = ExecutionEngine(paper_broker, RiskManager(settings), store)
    outcomes = engine.process([_signal()], {"AAA": 100.0})
    assert outcomes[0].action == "submitted"
    assert len(paper_broker.submitted) == 1


def test_live_broker_NEVER_auto_submits(settings, live_broker, store):
    """The core safety invariant: process() must not submit to a live broker."""
    engine = ExecutionEngine(live_broker, RiskManager(settings), store)
    outcomes = engine.process([_signal()], {"AAA": 100.0})
    assert outcomes[0].action == "queued"
    assert live_broker.submitted == []  # nothing sent
    assert store.list_pending()  # but it was queued for approval


def test_rejected_signal_reaches_neither_broker_nor_queue(settings, live_broker, store):
    settings.max_trade_notional = 1.0  # force a rejection
    engine = ExecutionEngine(live_broker, RiskManager(settings), store)
    outcomes = engine.process([_signal()], {"AAA": 100.0})
    assert outcomes[0].action == "rejected"
    assert live_broker.submitted == []
    assert store.list_pending() == []


def test_hold_and_missing_price_are_skipped(settings, paper_broker, store):
    engine = ExecutionEngine(paper_broker, RiskManager(settings), store)
    outcomes = engine.process(
        [_signal(t=SignalType.HOLD), _signal(symbol="ZZZ")], {"AAA": 100.0}
    )
    assert outcomes[0].action == "skipped"
    assert outcomes[1].action == "skipped"  # no price for ZZZ
    assert paper_broker.submitted == []
