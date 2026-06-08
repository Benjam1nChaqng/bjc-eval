"""Alpaca-backed Broker.

``paper=True`` hits the simulated sandbox; ``paper=False`` is real money.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from .broker import Account, OrderRecord, OrderRequest, OrderSide, Position


class AlpacaBroker:
    def __init__(self, api_key: str, secret: str, paper: bool) -> None:
        from alpaca.trading.client import TradingClient  # lazy import

        self.mode: Literal["paper", "live"] = "paper" if paper else "live"
        self._client = TradingClient(api_key, secret, paper=paper)

    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            blocked=bool(a.trading_blocked or a.account_blocked),
        )

    def get_positions(self) -> list[Position]:
        return [
            Position(symbol=p.symbol, qty=int(float(p.qty)), avg_price=float(p.avg_entry_price))
            for p in self._client.get_all_positions()
        ]

    def position_for(self, symbol: str) -> Position | None:
        return next((p for p in self.get_positions() if p.symbol == symbol), None)

    def submit_order(self, order: OrderRequest) -> OrderRecord:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL
        req = MarketOrderRequest(
            symbol=order.symbol, qty=order.qty, side=side, time_in_force=TimeInForce.DAY
        )
        result = self._client.submit_order(req)
        return OrderRecord(
            broker_order_id=str(result.id),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            submitted_at=datetime.utcnow(),
            status=str(result.status),
        )

    def has_withdrawal_scope(self) -> bool | None:
        """Best-effort: Alpaca trading keys can't move money, but we surface the check."""
        # The trading API exposes no transfer capability for these keys; report False.
        return False
