"""Shared fixtures: in-memory fake broker, deterministic bars, temp store/settings."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from bjc_trader.config import Settings, load_settings
from bjc_trader.execution.broker import (
    Account,
    OrderRecord,
    OrderRequest,
    Position,
)
from bjc_trader.store.db import Store


class FakeBroker:
    """In-memory Broker implementation for tests. Records every submitted order."""

    def __init__(
        self,
        mode: Literal["paper", "live"] = "paper",
        equity: float = 10_000.0,
        cash: float = 10_000.0,
        positions: list[Position] | None = None,
    ) -> None:
        self.mode = mode
        self._account = Account(equity=equity, cash=cash, buying_power=cash)
        self._positions = positions or []
        self.submitted: list[OrderRequest] = []

    def get_account(self) -> Account:
        return self._account

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def position_for(self, symbol: str) -> Position | None:
        return next((p for p in self._positions if p.symbol == symbol), None)

    def submit_order(self, order: OrderRequest) -> OrderRecord:
        self.submitted.append(order)
        return OrderRecord(
            broker_order_id=f"fake-{len(self.submitted)}",
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            submitted_at=datetime.utcnow(),
            status="accepted",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return load_settings(
        max_trade_notional=1000.0,
        max_position_pct=50.0,
        max_open_positions=5,
        max_daily_loss=100.0,
        db_path=str(tmp_path / "test.db"),
    )


@pytest.fixture
def store(settings: Settings) -> Store:
    return Store(settings.db_path)


@pytest.fixture
def paper_broker() -> FakeBroker:
    return FakeBroker(mode="paper")


@pytest.fixture
def live_broker() -> FakeBroker:
    return FakeBroker(mode="live")


def make_bars(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Build an OHLCV frame from a close-price series (O/H/L derived, flat volume)."""
    idx = pd.date_range(start=start, periods=len(closes), freq="D", tz="UTC")
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(closes), 1_000_000),
        },
        index=idx,
    )


@pytest.fixture
def trending_up_bars() -> pd.DataFrame:
    # A dip then a sustained rise — produces a clean MA crossover.
    return make_bars([10] * 30 + list(np.linspace(10, 30, 40)))
