"""Judge implementations for the swe-judge ensemble.

The Judge protocol is provider-agnostic. Concrete implementations wrap
Anthropic, OpenAI, and Google SDKs. A MockJudge is provided for fast,
deterministic, API-free testing.
"""

from swe_judge.judges.base import Judge, JudgeError
from swe_judge.judges.mock import MockJudge

__all__ = ["Judge", "JudgeError", "MockJudge"]
