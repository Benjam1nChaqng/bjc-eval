"""The human-approval gate for live (real-money) orders.

``approve`` is the ONLY function in the codebase that submits a live order. It:
  1. confirms the pending order is still PENDING_APPROVAL,
  2. re-runs the risk checks against *current* account state (state may have moved),
  3. refuses to submit while dry-run is on (prints intent instead),
  4. requires a confirmation callback (the CLI makes the human type the ticker).
"""

from __future__ import annotations

from dataclasses import dataclass

from bjc_trader.config import Settings
from bjc_trader.risk.manager import RiskManager
from bjc_trader.store.db import Store
from bjc_trader.store.models import PendingOrder, PendingStatus

from .broker import Broker, OrderRecord, OrderRequest, OrderSide


@dataclass
class ApprovalResult:
    ok: bool
    detail: str
    order_record: OrderRecord | None = None


class ApprovalGate:
    def __init__(self, store: Store, settings: Settings, risk: RiskManager) -> None:
        self.store = store
        self.settings = settings
        self.risk = risk

    def list_pending(self) -> list[PendingOrder]:
        return self.store.list_pending(only_open=True)

    def reject(self, order_id: str, reason: str = "manual reject") -> ApprovalResult:
        pending = self.store.get_pending(order_id)
        if pending is None or pending.status is not PendingStatus.PENDING_APPROVAL:
            return ApprovalResult(False, "no open pending order with that id")
        self.store.set_status(order_id, PendingStatus.REJECTED)
        return ApprovalResult(True, f"rejected ({reason})")

    def approve(
        self,
        order_id: str,
        live_broker: Broker,
        confirm: bool = False,
    ) -> ApprovalResult:
        """Approve and submit a live order. ``confirm`` must be True to actually submit."""
        if live_broker.mode != "live":
            return ApprovalResult(False, "approve requires a live broker")

        pending = self.store.get_pending(order_id)
        if pending is None or pending.status is not PendingStatus.PENDING_APPROVAL:
            return ApprovalResult(False, "no open pending order with that id")

        order = OrderRequest(
            symbol=pending.symbol,
            side=OrderSide(pending.side),
            qty=pending.qty,
            est_price=pending.est_price,
        )

        # Re-validate against fresh state — a lot can change between propose and approve.
        account = live_broker.get_account()
        positions = live_broker.get_positions()
        daily_pnl = self.store.daily_pnl()
        decision = self.risk.check(order, account, positions, daily_pnl)
        if not decision.approved:
            return ApprovalResult(False, "risk re-check failed: " + "; ".join(decision.reasons))

        if not confirm:
            return ApprovalResult(False, "not confirmed by operator")

        if self.settings.dry_run:
            return ApprovalResult(
                False,
                f"DRY-RUN: WOULD submit {order.side.value} {order.qty} {order.symbol} "
                f"(~${order.notional:.2f}). Set BJC_DRY_RUN=false to send for real.",
            )

        record = live_broker.submit_order(order)
        self.store.set_status(order_id, PendingStatus.SUBMITTED)
        return ApprovalResult(True, "submitted to live broker", record)
