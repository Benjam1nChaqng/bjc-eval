from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from bjc_trader import cli
from tests.conftest import make_bars

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    # Keep each test's sqlite + config out of the real environment.
    monkeypatch.setenv("BJC_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.delenv("BJC_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("BJC_LIVE_API_KEY", raising=False)


def test_doctor_reports_live_off():
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "live trading is OFF" in result.stdout


def test_pending_empty():
    result = runner.invoke(cli.app, ["pending"])
    assert result.exit_code == 0
    assert "Pending live orders" in result.stdout


def test_approve_unknown_id_exits_nonzero():
    result = runner.invoke(cli.app, ["approve", "deadbeef"])
    assert result.exit_code == 1
    assert "no pending order" in result.stdout


def test_backtest_runs_against_patched_provider(monkeypatch):
    class FakeProvider:
        def get_bars(self, symbol, start, end, timeframe="1d"):
            return make_bars([10] * 30 + list(np.linspace(10, 30, 40)))

    monkeypatch.setattr(cli, "_build_data_provider", lambda s, src: FakeProvider())
    result = runner.invoke(
        cli.app, ["backtest", "ma_crossover", "AAA", "--param", "fast=5", "--param", "slow=20"]
    )
    assert result.exit_code == 0
    assert "Backtest" in result.stdout
    assert "Total return" in result.stdout


def test_signals_command(monkeypatch):
    class FakeProvider:
        def get_bars(self, symbol, start, end, timeframe="1d"):
            return make_bars([10] * 30 + list(np.linspace(10, 30, 40)))

    monkeypatch.setattr(cli, "_build_data_provider", lambda s, src: FakeProvider())
    result = runner.invoke(
        cli.app, ["signals", "ma_crossover", "AAA", "--param", "fast=5", "--param", "slow=20"]
    )
    assert result.exit_code == 0
    assert "BUY" in result.stdout


def test_approve_confirmation_mismatch_submits_nothing(monkeypatch, tmp_path):
    # Queue a live order directly, then fail the typed-symbol confirmation.
    from bjc_trader.config import load_settings
    from bjc_trader.store.db import Store

    s = load_settings(db_path=str(tmp_path / "cli.db"))
    order = Store(s.db_path).add_pending("AAA", "BUY", 1, 100.0, "ma_crossover", "r")
    result = runner.invoke(cli.app, ["approve", order.id], input="WRONG\n")
    assert result.exit_code == 1
    assert "did not match" in result.stdout
