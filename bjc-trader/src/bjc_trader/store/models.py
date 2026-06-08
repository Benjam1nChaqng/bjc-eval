"""Row types for the persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class PendingStatus(str, Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"


@dataclass
class PendingOrder:
    id: str
    symbol: str
    side: str
    qty: int
    est_price: float
    strategy_name: str
    reason: str
    status: PendingStatus
    created_at: datetime
