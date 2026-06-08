"""Configuration and secrets handling.

The trading *mode* (paper vs live) and whether live trading is even possible are
derived strictly from which keys are present plus explicit enable flags — never
from a single free-floating boolean that could be flipped by accident.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["paper", "live"]


class Settings(BaseSettings):
    """Loaded from environment / .env with the ``BJC_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="BJC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paper (simulated) keys
    paper_api_key: SecretStr | None = None
    paper_secret: SecretStr | None = None

    # Live (real money) keys
    live_api_key: SecretStr | None = None
    live_secret: SecretStr | None = None

    # Safety flags — both default OFF.
    live_trading_enabled: bool = False
    dry_run: bool = True

    # Risk guardrails (all must be > 0).
    max_trade_notional: float = Field(default=100.0, gt=0)
    max_position_pct: float = Field(default=20.0, gt=0, le=100)
    max_open_positions: int = Field(default=5, gt=0)
    max_daily_loss: float = Field(default=50.0, gt=0)

    db_path: str = "./bjc.db"

    @model_validator(mode="after")
    def _check_key_pairs(self) -> Settings:
        # Keys come in pairs — reject a half-configured account loudly.
        if bool(self.paper_api_key) != bool(self.paper_secret):
            raise ValueError("paper_api_key and paper_secret must both be set or both empty")
        if bool(self.live_api_key) != bool(self.live_secret):
            raise ValueError("live_api_key and live_secret must both be set or both empty")
        return self

    @property
    def paper_keys_present(self) -> bool:
        return bool(self.paper_api_key and self.paper_secret)

    @property
    def live_keys_present(self) -> bool:
        return bool(self.live_api_key and self.live_secret)

    @property
    def live_enabled(self) -> bool:
        """Live trading is only possible with keys present AND the flag explicitly on."""
        return self.live_keys_present and self.live_trading_enabled

    def keys_for(self, mode: Mode) -> tuple[str, str]:
        """Return (api_key, secret) for the requested mode, or raise if unavailable."""
        if mode == "paper":
            if not self.paper_keys_present:
                raise ValueError("paper keys are not configured (set BJC_PAPER_API_KEY/SECRET)")
            assert self.paper_api_key and self.paper_secret
            return self.paper_api_key.get_secret_value(), self.paper_secret.get_secret_value()
        if not self.live_enabled:
            raise ValueError(
                "live trading is not enabled: set live keys and BJC_LIVE_TRADING_ENABLED=true"
            )
        assert self.live_api_key and self.live_secret
        return self.live_api_key.get_secret_value(), self.live_secret.get_secret_value()


def load_settings(**overrides: object) -> Settings:
    """Build Settings, applying explicit overrides (used by tests)."""
    return Settings(**overrides)
