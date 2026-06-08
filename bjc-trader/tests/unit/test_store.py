from __future__ import annotations

from datetime import date

from bjc_trader.store.db import Store
from bjc_trader.store.models import PendingStatus


def test_pending_orders_persist_across_restart(tmp_path):
    db = str(tmp_path / "p.db")
    s1 = Store(db)
    order = s1.add_pending("AAA", "BUY", 3, 100.0, "ma_crossover", "reason")
    # New Store instance == simulated process restart.
    s2 = Store(db)
    loaded = s2.get_pending(order.id)
    assert loaded is not None
    assert loaded.symbol == "AAA"
    assert loaded.status is PendingStatus.PENDING_APPROVAL


def test_status_transitions(tmp_path):
    s = Store(str(tmp_path / "p.db"))
    order = s.add_pending("AAA", "BUY", 1, 10.0, "x", "y")
    s.set_status(order.id, PendingStatus.SUBMITTED)
    assert s.get_pending(order.id).status is PendingStatus.SUBMITTED
    assert s.list_pending(only_open=True) == []


def test_daily_pnl_accumulates(tmp_path):
    s = Store(str(tmp_path / "p.db"))
    today = date(2026, 6, 8)
    assert s.daily_pnl(today) == 0.0
    s.record_realized(-30.0, today)
    s.record_realized(-20.0, today)
    assert s.daily_pnl(today) == -50.0


def test_schema_creation_idempotent(tmp_path):
    db = str(tmp_path / "p.db")
    Store(db)
    Store(db)  # second init must not raise
