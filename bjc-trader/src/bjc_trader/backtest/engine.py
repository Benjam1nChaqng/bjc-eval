"""A simple, deterministic long-only backtester driven by our pure Strategy interface.

It walks the bars once, applying each signal at the bar it fires on: BUY goes fully
long (cash -> shares at that close, minus commission), SELL exits to cash. This is
intentionally minimal and fully offline so results are reproducible in tests. It is
not a microstructure simulator — slippage beyond a flat commission is out of scope.
"""

from __future__ import annotations

import pandas as pd

from bjc_trader.strategies.base import SignalType, Strategy

from .metrics import (
    BacktestResult,
    cagr,
    max_drawdown,
    sharpe,
    total_return,
    win_rate,
)


def run_backtest(
    strategy: Strategy,
    symbol: str,
    bars: pd.DataFrame,
    cash: float = 1000.0,
    commission: float = 0.0,
) -> BacktestResult:
    signals = strategy.generate_signals(symbol, bars)
    # Map timestamp -> action for quick lookup as we walk the bars.
    action_at = {pd.Timestamp(s.timestamp): s.type for s in signals}

    equity_index: list[pd.Timestamp] = []
    equity_values: list[float] = []
    trade_returns: list[float] = []

    position_shares = 0.0
    entry_price = 0.0

    for ts, row in bars.iterrows():
        price = float(row["close"])
        ts = pd.Timestamp(ts)
        action = action_at.get(ts)

        if action is SignalType.BUY and position_shares == 0 and cash > 0:
            spend = cash * (1 - commission)
            position_shares = spend / price
            entry_price = price
            cash = 0.0
        elif action is SignalType.SELL and position_shares > 0:
            proceeds = position_shares * price * (1 - commission)
            trade_returns.append(price / entry_price - 1)
            cash = proceeds
            position_shares = 0.0

        equity_index.append(ts)
        equity_values.append(cash + position_shares * price)

    equity = pd.Series(equity_values, index=equity_index, dtype=float)
    return BacktestResult(
        total_return_pct=total_return(equity),
        cagr_pct=cagr(equity),
        sharpe=sharpe(equity),
        max_drawdown_pct=max_drawdown(equity),
        win_rate_pct=win_rate(trade_returns),
        num_trades=len(trade_returns),
        final_equity=float(equity.iloc[-1]) if not equity.empty else cash,
        equity_curve=equity,
    )
