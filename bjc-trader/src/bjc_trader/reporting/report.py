"""Render account state, pending orders, and backtest results as rich tables."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from bjc_trader.backtest.metrics import BacktestResult
from bjc_trader.execution.broker import Account, Position
from bjc_trader.store.models import PendingOrder


def positions_table(account: Account, positions: list[Position]) -> Table:
    table = Table(title=f"Account (equity ${account.equity:,.2f} · cash ${account.cash:,.2f})")
    table.add_column("Symbol")
    table.add_column("Qty", justify="right")
    table.add_column("Avg price", justify="right")
    for p in positions:
        table.add_row(p.symbol, str(p.qty), f"${p.avg_price:,.2f}")
    if not positions:
        table.add_row("—", "—", "—")
    return table


def pending_table(orders: list[PendingOrder]) -> Table:
    table = Table(title="Pending live orders (awaiting approval)")
    for col in ("ID", "Symbol", "Side", "Qty", "Est $", "Strategy", "Reason"):
        table.add_column(col)
    for o in orders:
        table.add_row(
            o.id,
            o.symbol,
            o.side,
            str(o.qty),
            f"{o.qty * o.est_price:,.2f}",
            o.strategy_name,
            o.reason,
        )
    if not orders:
        table.add_row("—", "—", "—", "—", "—", "—", "none")
    return table


def backtest_table(strategy_name: str, symbol: str, result: BacktestResult) -> Table:
    table = Table(title=f"Backtest · {strategy_name} · {symbol}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total return", f"{result.total_return_pct:,.2f}%")
    table.add_row("CAGR", f"{result.cagr_pct:,.2f}%")
    table.add_row("Sharpe", f"{result.sharpe:,.2f}")
    table.add_row("Max drawdown", f"{result.max_drawdown_pct:,.2f}%")
    table.add_row("Win rate", f"{result.win_rate_pct:,.1f}%")
    table.add_row("Trades", str(result.num_trades))
    table.add_row("Final equity", f"${result.final_equity:,.2f}")
    return table


def render(table: Table, console: Console | None = None) -> None:
    (console or Console()).print(table)
