"""Broker protocol and the order/account domain types.

The protocol is deliberately tiny. The live implementation and the in-memory test
fake both satisfy it, so nothing downstream knows or cares which it's talking to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Protocol, runtime_checkable


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    qty: int  # whole shares in v1
    est_price: float  # latest known price, for notional/risk estimation

    @property
    def notional(self) -> float:
        return self.qty * self.est_price


@dataclass(frozen=True)
class OrderRecord:
    broker_order_id: str
    symbol: str
    side: OrderSide
    qty: int
    submitted_at: datetime
    status: str


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    avg_price: float


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    blocked: bool = False


@runtime_checkable
class Broker(Protocol):
    mode: Literal["paper", "live"]

    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def position_for(self, symbol: str) -> Position | None: ...

    def submit_order(self, order: OrderRequest) -> OrderRecord: ...
