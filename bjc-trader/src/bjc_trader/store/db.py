"""SQLite persistence for the pending-order queue and the daily PnL ledger.

No ORM — small, explicit, and easy to audit. Schema creation is idempotent.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime

from .models import PendingOrder, PendingStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_orders (
    id            TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    qty           INTEGER NOT NULL,
    est_price     REAL NOT NULL,
    strategy_name TEXT NOT NULL,
    reason        TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_pnl (
    day      TEXT PRIMARY KEY,
    realized REAL NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- pending orders --------------------------------------------------
    def add_pending(
        self, symbol: str, side: str, qty: int, est_price: float, strategy_name: str, reason: str
    ) -> PendingOrder:
        order = PendingOrder(
            id=uuid.uuid4().hex[:12],
            symbol=symbol,
            side=side,
            qty=qty,
            est_price=est_price,
            strategy_name=strategy_name,
            reason=reason,
            status=PendingStatus.PENDING_APPROVAL,
            created_at=datetime.utcnow(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO pending_orders VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    order.id,
                    order.symbol,
                    order.side,
                    order.qty,
                    order.est_price,
                    order.strategy_name,
                    order.reason,
                    order.status.value,
                    order.created_at.isoformat(),
                ),
            )
        return order

    def get_pending(self, order_id: str) -> PendingOrder | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM pending_orders WHERE id = ?", (order_id,)
            ).fetchone()
        return _row_to_pending(row) if row else None

    def list_pending(self, only_open: bool = True) -> list[PendingOrder]:
        query = "SELECT * FROM pending_orders"
        params: tuple[object, ...] = ()
        if only_open:
            query += " WHERE status = ?"
            params = (PendingStatus.PENDING_APPROVAL.value,)
        query += " ORDER BY created_at"
        with self._conn() as c:
            rows = c.execute(query, params).fetchall()
        return [_row_to_pending(r) for r in rows]

    def set_status(self, order_id: str, status: PendingStatus) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE pending_orders SET status = ? WHERE id = ?", (status.value, order_id)
            )

    # --- daily pnl -------------------------------------------------------
    def record_realized(self, amount: float, when: date | None = None) -> float:
        day = (when or date.today()).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO daily_pnl(day, realized) VALUES (?, ?) "
                "ON CONFLICT(day) DO UPDATE SET realized = realized + excluded.realized",
                (day, amount),
            )
            row = c.execute("SELECT realized FROM daily_pnl WHERE day = ?", (day,)).fetchone()
        return float(row["realized"])

    def daily_pnl(self, when: date | None = None) -> float:
        day = (when or date.today()).isoformat()
        with self._conn() as c:
            row = c.execute("SELECT realized FROM daily_pnl WHERE day = ?", (day,)).fetchone()
        return float(row["realized"]) if row else 0.0


def _row_to_pending(row: sqlite3.Row) -> PendingOrder:
    return PendingOrder(
        id=row["id"],
        symbol=row["symbol"],
        side=row["side"],
        qty=row["qty"],
        est_price=row["est_price"],
        strategy_name=row["strategy_name"],
        reason=row["reason"],
        status=PendingStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
