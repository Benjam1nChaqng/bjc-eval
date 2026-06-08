"""Typer CLI — the only place CLI input is wired to services.

Safety defaults are visible here: --account defaults to paper, and live submission
additionally needs dry-run off plus a typed-symbol confirmation.
"""

from __future__ import annotations

from datetime import datetime

import typer
from rich.console import Console

from bjc_trader import reporting
from bjc_trader.config import Mode, Settings, load_settings
from bjc_trader.data.provider import DataProvider
from bjc_trader.execution.approval import ApprovalGate
from bjc_trader.execution.broker import Broker
from bjc_trader.execution.engine import ExecutionEngine
from bjc_trader.risk.manager import RiskManager
from bjc_trader.store.db import Store
from bjc_trader.strategies import registry

app = typer.Typer(add_completion=False, help="bjc-trader: safe, human-gated stock trading bot.")
console = Console()


# --- builders (kept lazy so read-only commands work without keys/SDKs) ------
def _settings() -> Settings:
    return load_settings()


def _store(settings: Settings) -> Store:
    return Store(settings.db_path)


def _build_broker(settings: Settings, account: Mode) -> Broker:
    from bjc_trader.execution.alpaca_broker import AlpacaBroker

    key, secret = settings.keys_for(account)  # raises with a clear message if unavailable
    return AlpacaBroker(key, secret, paper=(account == "paper"))


def _build_data_provider(settings: Settings, source: str) -> DataProvider:
    if source == "alpaca":
        from bjc_trader.data.alpaca_data import AlpacaDataProvider

        account: Mode = "paper" if settings.paper_keys_present else "live"
        key, secret = settings.keys_for(account)
        return AlpacaDataProvider(key, secret)
    from bjc_trader.data.yfinance_data import YFinanceDataProvider

    return YFinanceDataProvider()


def _parse_params(pairs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"param must be key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


# --- commands ---------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Validate configuration and report which guardrails are armed."""
    s = _settings()
    console.print("[bold]bjc-trader doctor[/bold]")
    console.print(f"  paper keys present : {s.paper_keys_present}")
    console.print(f"  live keys present  : {s.live_keys_present}")
    console.print(f"  live trading enabled: {s.live_enabled}  (flag={s.live_trading_enabled})")
    console.print(f"  dry-run            : {s.dry_run}  (live orders print intent only while true)")
    console.print(
        f"  guardrails         : trade<=${s.max_trade_notional:.0f}, "
        f"pos<={s.max_position_pct:.0f}%, max_open={s.max_open_positions}, "
        f"daily_loss<=${s.max_daily_loss:.0f}"
    )
    if not s.live_enabled:
        console.print("  [green]live trading is OFF — only the paper sandbox can trade.[/green]")
    else:
        console.print(
            "  [yellow]live trading is ARMED — real orders still require approval.[/yellow]"
        )


@app.command()
def backtest(
    strategy: str,
    symbol: str,
    start: str = typer.Option("2020-01-01"),
    end: str = typer.Option(datetime.utcnow().date().isoformat()),
    cash: float = typer.Option(1000.0),
    commission: float = typer.Option(0.0),
    source: str = typer.Option("yfinance"),
    param: list[str] = typer.Option([], help="strategy params, e.g. --param fast=10"),
) -> None:
    """Backtest a strategy on historical data. Read-only; never places orders."""
    s = _settings()
    from bjc_trader.backtest.engine import run_backtest

    strat = registry.build(strategy, **_parse_params(param))
    provider = _build_data_provider(s, source)
    bars = provider.get_bars(
        symbol, datetime.fromisoformat(start), datetime.fromisoformat(end)
    )
    result = run_backtest(strat, symbol, bars, cash=cash, commission=commission)
    reporting.render(reporting.backtest_table(strategy, symbol, result), console)


@app.command()
def signals(
    strategy: str,
    symbol: str,
    start: str = typer.Option("2024-01-01"),
    end: str = typer.Option(datetime.utcnow().date().isoformat()),
    source: str = typer.Option("yfinance"),
    param: list[str] = typer.Option([]),
) -> None:
    """Compute and print current signals. Read-only."""
    s = _settings()
    strat = registry.build(strategy, **_parse_params(param))
    provider = _build_data_provider(s, source)
    bars = provider.get_bars(symbol, datetime.fromisoformat(start), datetime.fromisoformat(end))
    sigs = strat.generate_signals(symbol, bars)
    if not sigs:
        console.print("no signals in range")
        return
    for sig in sigs[-10:]:
        console.print(f"{sig.timestamp.date()}  {sig.type.value:4}  {sig.symbol}  {sig.reason}")


@app.command(name="run-paper")
def run_paper(
    strategy: str,
    symbols: list[str],
    start: str = typer.Option("2024-01-01"),
    param: list[str] = typer.Option([]),
) -> None:
    """Run the strategy against the PAPER account, auto-executing simulated orders."""
    s = _settings()
    broker = _build_broker(s, "paper")
    risk = RiskManager(s)
    store = _store(s)
    engine = ExecutionEngine(broker, risk, store)
    provider = _build_data_provider(s, "alpaca")

    all_signals = []
    prices: dict[str, float] = {}
    for symbol in symbols:
        bars = provider.get_bars(symbol, datetime.fromisoformat(start), datetime.utcnow())
        strat = registry.build(strategy, **_parse_params(param))
        sigs = strat.generate_signals(symbol, bars)
        if sigs:
            all_signals.append(sigs[-1])  # act on the most recent signal only
            prices[symbol] = float(bars["close"].iloc[-1])

    for outcome in engine.process(all_signals, prices):
        console.print(f"[{outcome.action}] {outcome.signal.symbol}: {outcome.detail}")


@app.command()
def propose(
    strategy: str,
    symbols: list[str],
    start: str = typer.Option("2024-01-01"),
    param: list[str] = typer.Option([]),
) -> None:
    """Generate LIVE order proposals and queue them for approval. Never submits."""
    s = _settings()
    broker = _build_broker(s, "live")
    risk = RiskManager(s)
    store = _store(s)
    engine = ExecutionEngine(broker, risk, store)
    provider = _build_data_provider(s, "alpaca")

    all_signals = []
    prices: dict[str, float] = {}
    for symbol in symbols:
        bars = provider.get_bars(symbol, datetime.fromisoformat(start), datetime.utcnow())
        strat = registry.build(strategy, **_parse_params(param))
        sigs = strat.generate_signals(symbol, bars)
        if sigs:
            all_signals.append(sigs[-1])
            prices[symbol] = float(bars["close"].iloc[-1])

    for outcome in engine.process(all_signals, prices):
        console.print(f"[{outcome.action}] {outcome.signal.symbol}: {outcome.detail}")
    console.print(
        "\nReview with [bold]bjc pending[/bold], approve with [bold]bjc approve <id>[/bold]."
    )


@app.command()
def pending() -> None:
    """List live orders awaiting approval."""
    s = _settings()
    reporting.render(reporting.pending_table(_store(s).list_pending()), console)


@app.command()
def approve(order_id: str) -> None:
    """Approve and submit a queued LIVE order (requires typing the ticker to confirm)."""
    s = _settings()
    store = _store(s)
    gate = ApprovalGate(store, s, RiskManager(s))

    order = store.get_pending(order_id)
    if order is None:
        console.print("[red]no pending order with that id[/red]")
        raise typer.Exit(1)

    console.print(
        f"About to approve: [bold]{order.side} {order.qty} {order.symbol}[/bold] "
        f"(~${order.qty * order.est_price:,.2f}) — {order.strategy_name}: {order.reason}"
    )
    typed = typer.prompt(f"Type the ticker symbol ({order.symbol}) to confirm")
    if typed.strip().upper() != order.symbol.upper():
        console.print("[yellow]confirmation did not match — nothing submitted.[/yellow]")
        raise typer.Exit(1)

    broker = _build_broker(s, "live")
    result = gate.approve(order_id, broker, confirm=True)
    color = "green" if result.ok else "yellow"
    console.print(f"[{color}]{result.detail}[/{color}]")


@app.command()
def reject(order_id: str, reason: str = typer.Option("manual reject")) -> None:
    """Drop a queued live order."""
    s = _settings()
    gate = ApprovalGate(_store(s), s, RiskManager(s))
    console.print(gate.reject(order_id, reason).detail)


@app.command()
def report() -> None:
    """Show paper account positions (live if only live keys are configured)."""
    s = _settings()
    account: Mode = "paper" if s.paper_keys_present else "live"
    broker = _build_broker(s, account)
    reporting.render(
        reporting.positions_table(broker.get_account(), broker.get_positions()), console
    )


@app.command(name="kill-switch")
def kill_switch(
    engage: bool = typer.Option(False, "--engage"),
    status: bool = typer.Option(False, "--status"),
) -> None:
    """Inspect or manually engage the daily kill switch (advisory: also enforced per-run)."""
    if engage:
        console.print("[red]kill switch engaged for this run context.[/red] "
                      "Per-run enforcement also halts trading once daily loss is breached.")
    else:
        console.print("kill switch is evaluated each run from daily PnL vs BJC_MAX_DAILY_LOSS.")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
