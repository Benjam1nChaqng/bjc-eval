"""Rubric definitions for swe-judge.

Each rubric version is in its own module (v1.py, v2.py, ...). Version is
captured in the Run.config so historical scores remain interpretable as
the rubric evolves.
"""

from swe_judge.rubrics.v1 import RUBRIC_V1, RubricDimension

__all__ = ["RUBRIC_V1", "RubricDimension"]
