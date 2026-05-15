"""Tests for the v1 rubric."""

from __future__ import annotations

from swe_judge.rubrics.v1 import (
    CODE_QUALITY,
    CORRECTNESS,
    REASONING,
    RUBRIC_V1,
    rubric_v1_full_text,
)


class TestRubricV1:
    def test_has_all_three_dimensions(self) -> None:
        assert set(RUBRIC_V1.keys()) == {"correctness", "code_quality", "reasoning"}

    def test_each_dimension_has_five_anchors(self) -> None:
        for dim in RUBRIC_V1.values():
            assert set(dim.anchors.keys()) == {1, 2, 3, 4, 5}

    def test_anchors_are_nonempty_strings(self) -> None:
        for dim in RUBRIC_V1.values():
            for score, text in dim.anchors.items():
                assert isinstance(text, str)
                assert len(text) > 20, f"{dim.name} anchor {score} is too short"

    def test_anchor_table_includes_dimension_name(self) -> None:
        rendered = CORRECTNESS.anchor_table()
        assert "Correctness" in rendered
        assert "5" in rendered
        assert "1" in rendered

    def test_full_text_includes_all_three(self) -> None:
        text = rubric_v1_full_text()
        assert "Correctness" in text
        assert "Code Quality" in text
        assert "Reasoning" in text

    def test_dimension_consistency(self) -> None:
        """The .dimension field must match the dict key."""
        for key, dim in RUBRIC_V1.items():
            assert dim.dimension == key

    def test_anchors_are_ordered_top_down(self) -> None:
        """anchor_table() should list 5 first, 1 last."""
        rendered = CORRECTNESS.anchor_table()
        idx_5 = rendered.index("**5**")
        idx_1 = rendered.index("**1**")
        assert idx_5 < idx_1
