"""Turn signals into orders — safely.

Paper brokers auto-execute. Live brokers NEVER auto-execute here: there is no code
path in this module that calls ``submit_order`` on a live broker. Live orders that
pass risk become PENDING_APPROVAL rows for a human to approve. A test asserts the
live broker's submission list stays empty after ``process``.
"""

from __future__ import annotations

from dataclasses import dataclass

from bjc_trader.risk.manager import RiskManager
from bjc_trader.store.db import Store
from bjc_trader.strategies.base import Signal, SignalType

from .broker import Broker, OrderRecord


@dataclass
class ExecutionOutcome:
    signal: Signal
    action: str  # "submitted" | "queued" | "rejected" | "skipped"
    detail: str
    order_record: OrderRecord | None = None
    pending_id: str | None = None


class ExecutionEngine:
    def __init__(self, broker: Broker, risk: RiskManager, store: Store) -> None:
        self.broker = broker
        self.risk = risk
        self.store = store

    def process(self, signals: list[Signal], prices: dict[str, float]) -> list[ExecutionOutcome]:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        daily_pnl = self.store.daily_pnl()
        outcomes: list[ExecutionOutcome] = []

        for signal in signals:
            if signal.type is SignalType.HOLD:
                outcomes.append(ExecutionOutcome(signal, "skipped", "HOLD"))
                continue
            price = prices.get(signal.symbol)
            if not price or price <= 0:
                outcomes.append(ExecutionOutcome(signal, "skipped", "no price available"))
                continue

            order = self.risk.size_order(signal, account, price)
            decision = self.risk.check(order, account, positions, daily_pnl)
            if not decision.approved:
                outcomes.append(
                    ExecutionOutcome(signal, "rejected", "; ".join(decision.reasons))
                )
                continue

            if self.broker.mode == "paper":
                record = self.broker.submit_order(order)
                outcomes.append(
                    ExecutionOutcome(signal, "submitted", "paper auto-execute", record)
                )
            else:
                # LIVE: queue only. We never submit here.
                pending = self.store.add_pending(
                    symbol=order.symbol,
                    side=order.side.value,
                    qty=order.qty,
                    est_price=order.est_price,
                    strategy_name=signal.strategy_name,
                    reason=signal.reason,
                )
                outcomes.append(
                    ExecutionOutcome(
                        signal,
                        "queued",
                        "live order requires human approval (`bjc approve`)",
                        pending_id=pending.id,
                    )
                )
        return outcomes
