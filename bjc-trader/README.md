# bjc-trader

A safe, human-gated stock trading bot. It fetches market data, computes signals from
classic strategies, backtests them on historical data, **auto-executes only in a simulated
("paper") account**, and treats every real-money order as a recommendation you must
explicitly approve.

> ## ⚠️ Risk & disclaimer — read this first
>
> - **This is educational software, not financial advice.** The author is not a licensed
>   financial advisor.
> - **Trading stocks can lose money, up to your entire balance.** The research is one-sided:
>   roughly **70% of retail traders lose money** using automated bots, and **80%+ of fully
>   automated retail bots underperform** simply buying and holding a low-cost index fund after
>   fees and slippage. ~95% of day traders lose money in their first year.
> - Even well-tested strategies realistically return **single digits to low-teens %** a year.
>   If a bot promises more, it is lying or hiding the drawdowns.
> - **You are responsible for every order you approve.** There is no warranty.
> - **Recommended:** keep the bulk of your money compounding in a low-cost, broad-market index
>   fund (e.g. an S&P 500 ETF, dollar-cost-averaged). Run this bot on **paper** first — for
>   weeks or months — and only ever risk money you can afford to lose entirely.

## What it is / what it is NOT

| It is | It is NOT |
|-------|-----------|
| A backtesting + paper-trading + signal engine | A money printer or "passive income" machine |
| A human-in-the-loop advisor for real orders | An autonomous real-money trader |
| Built on reputable, audited libraries | A copy-paste of a "guaranteed profit" bot |

## Safety model — four independent locks on live orders

A real-money order is structurally impossible to send by accident. **All four** defaults must
be deliberately changed:

1. **Key separation** — the live broker is only constructed when both `BJC_LIVE_API_KEY`/
   `BJC_LIVE_SECRET` are set **and** `BJC_LIVE_TRADING_ENABLED=true`. Generate live keys as
   **trading-only (no withdrawal/banking scope)**.
2. **The engine cannot auto-submit live** — `ExecutionEngine.process()` has *no code path* that
   submits to a live broker; live signals can only become `PENDING_APPROVAL` rows. A test
   (`test_live_broker_NEVER_auto_submits`) asserts this invariant.
3. **Explicit human approval** — `bjc approve <id>` is the *only* function that submits a live
   order. It re-runs risk checks against current account state and makes you **type the ticker
   symbol** to confirm (not a bare y/N).
4. **Dry-run by default** — while `BJC_DRY_RUN=true`, approval prints "WOULD submit" and sends
   nothing. Real submission requires explicitly setting `BJC_DRY_RUN=false`.

**Guardrails (`RiskManager`):** per-trade notional cap, max position % of equity, max open
positions, **no shorting**, **no margin** (orders must fit settled cash), no options, and a
**daily-loss kill switch** that halts trading once breached.

## Install

```bash
python -m pip install -e ".[dev]"        # core + test tooling
# optional extras when you connect a broker / pull live data / use the bundled backtester:
python -m pip install -e ".[broker,data]"
```

Requires Python 3.11+.

## Configure

```bash
cp .env.example .env       # .env is gitignored — never commit it
```

Create free Alpaca **paper** keys at <https://alpaca.markets/> and put them in `.env`. Leave
the live keys empty until you have validated a strategy on paper for a meaningful period.

```bash
bjc doctor      # shows active mode and which guardrails are armed
```

## Quickstart (no real money)

```bash
# 1. Backtest a strategy on years of history (read-only, no orders)
bjc backtest ma_crossover SPY --start 2020-01-01 --end 2024-12-31 --param fast=20 --param slow=50

# 2. See the latest signals
bjc signals rsi_reversion AAPL

# 3. Auto-execute in the PAPER sandbox (needs paper keys)
bjc run-paper ma_crossover AAPL --param fast=20 --param slow=50
bjc report
```

## Going live (advisor mode)

Real orders are never automatic. The flow is propose → review → approve:

```bash
bjc propose ma_crossover AAPL          # queues a proposal; submits nothing
bjc pending                            # review what's queued
bjc approve <order-id>                 # type the ticker to confirm; dry-run prints intent only
```

To actually transmit a real order you must: set live keys, `BJC_LIVE_TRADING_ENABLED=true`,
`BJC_DRY_RUN=false`, run `approve`, and pass the typed-symbol confirmation. Four deliberate steps.

## Strategies

Strategies are pure functions over a price DataFrame and live in `src/bjc_trader/strategies/`.
v1 ships:

- `ma_crossover` — moving-average crossover (params: `fast`, `slow`).
- `rsi_reversion` — RSI mean-reversion (params: `period`, `oversold`, `overbought`).

Add one by implementing the `Strategy` protocol (`generate_signals(symbol, bars) -> list[Signal]`)
and registering it in `strategies/registry.py`. Strategies never size positions or place orders —
that keeps them trivial to test and safe to backtest.

## Architecture

```
signal (strategy)  ->  size + risk check (RiskManager)  ->  ExecutionEngine
                                                              ├─ paper: submit immediately
                                                              └─ live : enqueue PENDING_APPROVAL
                                                                          └─ bjc approve -> ApprovalGate -> live broker
```

| Layer | Module |
|-------|--------|
| Config & secrets | `config.py` |
| Data providers | `data/` (yfinance for backtests, Alpaca for live) |
| Strategies | `strategies/` |
| Backtester + metrics | `backtest/` |
| Risk guardrails | `risk/manager.py` |
| Execution + approval | `execution/` |
| Persistence (SQLite) | `store/` |
| CLI | `cli.py` |

## Testing

```bash
pytest          # 49 tests, all offline (no API keys, no network)
ruff check src tests
mypy
```

The whole suite runs against an in-memory fake broker, so it is green without any keys.

## License

MIT — see [LICENSE](LICENSE).
