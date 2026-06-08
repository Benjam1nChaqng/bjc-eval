"""Risk guardrails and position sizing.

Everything here is pure: given an order, account, positions and the day's PnL, decide
yes/no with reasons. No I/O, no broker. The kill switch is the one piece of mutable
state and it only ever moves toward "more restrictive" within a trading day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bjc_trader.config import Settings
from bjc_trader.execution.broker import Account, OrderRequest, OrderSide, Position
from bjc_trader.strategies.base import Signal, SignalType


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._kill_switch = False

    # --- kill switch -----------------------------------------------------
    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch

    def engage_kill_switch(self) -> None:
        self._kill_switch = True

    def release_kill_switch(self) -> None:
        self._kill_switch = False

    def update_daily_pnl(self, daily_pnl: float) -> None:
        """Engage the kill switch once the daily loss limit is breached."""
        if daily_pnl <= -abs(self.s.max_daily_loss):
            self._kill_switch = True

    # --- sizing ----------------------------------------------------------
    def size_order(self, signal: Signal, account: Account, price: float) -> OrderRequest:
        """Fixed-fractional sizing, capped by the per-trade notional limit."""
        if price <= 0:
            raise ValueError("price must be positive to size an order")
        side = OrderSide.BUY if signal.type is SignalType.BUY else OrderSide.SELL
        budget = min(self.s.max_trade_notional, account.equity * self.s.max_position_pct / 100)
        qty = int(budget // price)
        return OrderRequest(symbol=signal.symbol, side=side, qty=qty, est_price=price)

    # --- the gate --------------------------------------------------------
    def check(
        self,
        order: OrderRequest,
        account: Account,
        positions: list[Position],
        daily_pnl: float = 0.0,
    ) -> RiskDecision:
        reasons: list[str] = []

        # Daily loss first — breaching it both rejects and latches the kill switch.
        self.update_daily_pnl(daily_pnl)
        if self._kill_switch:
            return RiskDecision(False, ["kill switch engaged (daily loss limit or manual)"])

        if account.blocked:
            reasons.append("account is blocked by the broker")
        if order.qty <= 0:
            reasons.append("order quantity must be positive")

        held = next((p.qty for p in positions if p.symbol == order.symbol), 0)

        if order.side is OrderSide.SELL:
            # No shorting: never sell more than currently held.
            if order.qty > held:
                reasons.append(f"would short {order.symbol} (sell {order.qty} > held {held})")
        else:  # BUY
            # No margin: notional must fit within settled cash.
            if order.notional > account.cash:
                reasons.append(
                    f"insufficient cash: notional {order.notional:.2f} > cash {account.cash:.2f}"
                )
            if order.notional > self.s.max_trade_notional:
                reasons.append(
                    f"notional {order.notional:.2f} exceeds per-trade cap "
                    f"{self.s.max_trade_notional:.2f}"
                )
            # Position concentration cap.
            new_position_value = (held + order.qty) * order.est_price
            max_value = account.equity * self.s.max_position_pct / 100
            if new_position_value > max_value:
                reasons.append(
                    f"position {order.symbol} would be {new_position_value:.2f}, "
                    f"over {self.s.max_position_pct:.0f}% equity cap {max_value:.2f}"
                )
            # Max open positions (only if this opens a brand-new position).
            if held == 0:
                open_count = sum(1 for p in positions if p.qty != 0)
                if open_count >= self.s.max_open_positions:
                    reasons.append(
                        f"already at max open positions ({self.s.max_open_positions})"
                    )

        return RiskDecision(len(reasons) == 0, reasons)
