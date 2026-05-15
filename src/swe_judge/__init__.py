"""swe-judge: Multi-judge LLM evaluation harness for software-engineering tasks."""

from swe_judge.tasks import (
    Category,
    Dimension,
    HumanScore,
    Run,
    Score,
    Task,
    TestCase,
)

__version__ = "0.1.0"

__all__ = [
    "Category",
    "Dimension",
    "HumanScore",
    "Run",
    "Score",
    "Task",
    "TestCase",
    "__version__",
]
