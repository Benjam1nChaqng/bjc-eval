from __future__ import annotations

import pytest

from bjc_trader.config import load_settings


def test_defaults_are_safe():
    s = load_settings()
    assert s.dry_run is True
    assert s.live_trading_enabled is False
    assert s.live_enabled is False


def test_paper_keys_enable_paper_only():
    s = load_settings(paper_api_key="k", paper_secret="x")
    assert s.paper_keys_present is True
    assert s.live_enabled is False
    assert s.keys_for("paper") == ("k", "x")


def test_live_keys_without_flag_stay_disabled():
    s = load_settings(live_api_key="k", live_secret="x")
    assert s.live_keys_present is True
    assert s.live_enabled is False
    with pytest.raises(ValueError, match="live trading is not enabled"):
        s.keys_for("live")


def test_live_enabled_requires_keys_and_flag():
    s = load_settings(live_api_key="k", live_secret="x", live_trading_enabled=True)
    assert s.live_enabled is True
    assert s.keys_for("live") == ("k", "x")


def test_half_configured_keys_rejected():
    with pytest.raises(ValueError, match="must both be set"):
        load_settings(paper_api_key="k")


def test_guardrails_must_be_positive():
    with pytest.raises(ValueError):
        load_settings(max_trade_notional=0)
