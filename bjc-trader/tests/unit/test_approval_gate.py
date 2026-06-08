from __future__ import annotations

from bjc_trader.config import load_settings
from bjc_trader.execution.approval import ApprovalGate
from bjc_trader.risk.manager import RiskManager
from bjc_trader.store.models import PendingStatus


def _gate(store, settings):
    return ApprovalGate(store, settings, RiskManager(settings))


def _queue_one(store, symbol="AAA", qty=1, price=100.0):
    return store.add_pending(symbol, "BUY", qty, price, "ma_crossover", "reason")


def test_approve_without_confirm_does_not_submit(settings, store, live_broker):
    gate = _gate(store, settings)
    pending = _queue_one(store)
    result = gate.approve(pending.id, live_broker, confirm=False)
    assert not result.ok
    assert live_broker.submitted == []
    assert store.get_pending(pending.id).status is PendingStatus.PENDING_APPROVAL


def test_dry_run_blocks_submission_even_when_confirmed(tmp_path, store, live_broker):
    s = load_settings(dry_run=True, max_trade_notional=1000.0, db_path=str(tmp_path / "d.db"))
    gate = _gate(store, s)
    pending = _queue_one(store)
    result = gate.approve(pending.id, live_broker, confirm=True)
    assert not result.ok
    assert "DRY-RUN" in result.detail
    assert live_broker.submitted == []


def test_confirmed_non_dry_run_submits_exactly_once(tmp_path, store, live_broker):
    s = load_settings(dry_run=False, max_trade_notional=1000.0, db_path=str(tmp_path / "d.db"))
    gate = _gate(store, s)
    pending = _queue_one(store)
    result = gate.approve(pending.id, live_broker, confirm=True)
    assert result.ok
    assert len(live_broker.submitted) == 1
    assert store.get_pending(pending.id).status is PendingStatus.SUBMITTED


def test_risk_recheck_can_still_reject_at_approval(tmp_path, store, live_broker):
    s = load_settings(dry_run=False, max_trade_notional=1000.0, db_path=str(tmp_path / "d.db"))
    gate = _gate(store, s)
    # Queue an order that exceeds current cash (live broker only has 10k but qty huge).
    pending = _queue_one(store, qty=1000, price=100.0)  # notional 100k > cash
    result = gate.approve(pending.id, live_broker, confirm=True)
    assert not result.ok
    assert "risk re-check failed" in result.detail
    assert live_broker.submitted == []


def test_approving_non_live_broker_refused(settings, store, paper_broker):
    gate = _gate(store, settings)
    pending = _queue_one(store)
    result = gate.approve(pending.id, paper_broker, confirm=True)
    assert not result.ok
    assert "live broker" in result.detail


def test_reject_removes_from_open_queue(settings, store):
    gate = _gate(store, settings)
    pending = _queue_one(store)
    assert gate.reject(pending.id).ok
    assert store.get_pending(pending.id).status is PendingStatus.REJECTED
    assert store.list_pending(only_open=True) == []
