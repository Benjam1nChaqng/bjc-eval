"""Rubric v1 — three dimensions, worked anchors at every score level.

Design principles:
1. Every score level has a *worked anchor* — concrete language a judge can
   match against, not just a number.
2. Dimensions are independent — a model can score 5/2/3 across the three.
3. Rationale + anchor citation is *required* of every judge to enforce
   "evidence-first" scoring (judge must say why, then assign the number).

This rubric is intentionally not Claude- or GPT-specific. It is the same
text for every judge in the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

from swe_judge.tasks import Dimension


@dataclass(frozen=True)
class RubricDimension:
    """One dimension of the rubric, with its 5 anchored score levels."""

    dimension: Dimension
    name: str
    description: str
    anchors: dict[int, str]  # score -> anchor text

    def anchor_table(self) -> str:
        """Format anchors as a human/LLM-readable table for prompt inclusion."""
        lines = [f"### {self.name} — {self.description}", ""]
        for score in sorted(self.anchors.keys(), reverse=True):
            lines.append(f"- **{score}** — {self.anchors[score]}")
        return "\n".join(lines)


CORRECTNESS = RubricDimension(
    dimension="correctness",
    name="Correctness",
    description="Does the code do what was asked, including on edge cases?",
    anchors={
        5: (
            "Perfect. All test cases pass. No edge-case failures discoverable on "
            "inspection. The solution handles empty inputs, off-by-one boundaries, "
            "and type edge cases as appropriate for the prompt."
        ),
        4: (
            "Strong. All explicit test cases pass. A minor edge case may be "
            "unhandled but the code does not crash — e.g. empty input returns a "
            "silently wrong value but does not raise."
        ),
        3: (
            "Functional. Primary test cases pass. One or more edge cases fail "
            "outright or produce clearly wrong output."
        ),
        2: (
            "Partial. Approach is right but implementation has bugs that break "
            "multiple test cases. Symptoms suggest the author misunderstood part "
            "of the spec, not all of it."
        ),
        1: (
            "Broken. Does not run, raises on the happy path, or fundamentally "
            "misunderstands the task."
        ),
    },
)


CODE_QUALITY = RubricDimension(
    dimension="code_quality",
    name="Code Quality",
    description="Idiomatic, readable, appropriately abstracted code.",
    anchors={
        5: (
            "Production-grade. Idiomatic for the language. Well-named variables "
            "and functions. Follows community conventions (PEP 8 for Python). "
            "Abstractions are sized correctly — no over-engineering, no copy-paste."
        ),
        4: (
            "Solid. Readable and conventional. Minor style issues — inconsistent "
            "naming style, a missing docstring, a slightly awkward expression — "
            "but nothing a code reviewer would block on."
        ),
        3: (
            "Acceptable. Works, but smells: unclear names like `x` or `tmp`, "
            "inconsistent style within the file, light over-engineering (a class "
            "where a function would do), or obvious helper extractions missed."
        ),
        2: (
            "Poor. Significant readability or maintainability problems — magic "
            "numbers without context, repeated code instead of a loop, dead code, "
            "tightly coupled side effects."
        ),
        1: (
            "Unreadable. Intent is difficult to determine without running the code. "
            "Variable shadowing, deeply nested logic without comments, or wildly "
            "non-idiomatic style for the language."
        ),
    },
)


REASONING = RubricDimension(
    dimension="reasoning",
    name="Reasoning",
    description="Quality of the explanation accompanying the code or review.",
    anchors={
        5: (
            "Exemplary. Surfaces non-obvious tradeoffs. Anticipates edge cases "
            "the prompt did not explicitly mention. References relevant principles "
            "(e.g. 'this is a classic off-by-one because Python slice end is "
            "exclusive') or prior art when appropriate."
        ),
        4: (
            "Clear. Explanation is complete and correct for what the code does. "
            "Standard reasoning quality from a competent engineer — no errors but "
            "no insight beyond the obvious."
        ),
        3: (
            "Adequate. Covers the basics — what changed and why. Missing context "
            "an experienced reviewer would add (e.g. doesn't mention why the bug "
            "existed in the first place, doesn't flag related fragile code)."
        ),
        2: (
            "Surface-level. Restates what the code does without explaining why "
            "the chosen approach is correct or what alternatives were considered."
        ),
        1: (
            "Absent or wrong. No explanation provided, or the explanation directly "
            "contradicts what the code actually does."
        ),
    },
)


RUBRIC_V1: dict[Dimension, RubricDimension] = {
    "correctness": CORRECTNESS,
    "code_quality": CODE_QUALITY,
    "reasoning": REASONING,
}
"""The canonical v1 rubric, keyed by Dimension."""


def rubric_v1_full_text() -> str:
    """Render the full v1 rubric as markdown for inclusion in judge prompts."""
    sections = [r.anchor_table() for r in RUBRIC_V1.values()]
    return "\n\n".join(sections)
