"""Tests for the reliability module."""

from __future__ import annotations

import math

import pytest

from swe_judge.reliability import (
    cohen_kappa_pair,
    inter_judge_agreement,
    judge_vs_human,
    krippendorff_alpha,
    mean_inter_judge_kappa,
    mean_judge_vs_human,
    summary,
)
from swe_judge.tasks import HumanScore, Score


class TestCohenKappa:
    def test_perfect_agreement_kappa_is_one(self, perfect_agreement_scores: list[Score]) -> None:
        kappa = mean_inter_judge_kappa(perfect_agreement_scores)
        assert kappa == pytest.approx(1.0)

    def test_disagreement_kappa_less_than_one(self, disagreement_scores: list[Score]) -> None:
        kappa = mean_inter_judge_kappa(disagreement_scores)
        assert kappa < 1.0
        assert kappa > 0.0  # still some agreement

    def test_inter_judge_agreement_pairwise_keys_sorted(
        self, perfect_agreement_scores: list[Score]
    ) -> None:
        pairs = inter_judge_agreement(perfect_agreement_scores)
        for ja, jb in pairs.keys():
            assert ja < jb

    def test_cohen_kappa_pair_too_few_observations(self) -> None:
        from swe_judge.tasks import Score as S

        a = [S(run_id="R", task_id="t1", judge_model="A",
               dimension="correctness", value=4, rationale="x", anchor_matched="x")]
        b = [S(run_id="R", task_id="t1", judge_model="B",
               dimension="correctness", value=4, rationale="x", anchor_matched="x")]
        with pytest.raises(ValueError, match="at least 2"):
            cohen_kappa_pair(a, b)


class TestJudgeVsHuman:
    def test_perfect_human_alignment(
        self,
        perfect_agreement_scores: list[Score],
        human_scores_for_agreement: list[HumanScore],
    ) -> None:
        per_judge = judge_vs_human(perfect_agreement_scores, human_scores_for_agreement)
        assert set(per_judge.keys()) == {"judge-A", "judge-B"}
        for kappa in per_judge.values():
            assert kappa == pytest.approx(1.0)

    def test_mean_judge_vs_human(
        self,
        perfect_agreement_scores: list[Score],
        human_scores_for_agreement: list[HumanScore],
    ) -> None:
        m = mean_judge_vs_human(perfect_agreement_scores, human_scores_for_agreement)
        assert m == pytest.approx(1.0)

    def test_mean_judge_vs_human_empty_returns_nan(self) -> None:
        result = mean_judge_vs_human([], [])
        assert math.isnan(result)


class TestKrippendorffAlpha:
    def test_perfect_agreement_alpha_close_to_one(
        self, perfect_agreement_scores: list[Score]
    ) -> None:
        alpha = krippendorff_alpha(perfect_agreement_scores)
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_with_human_scorer_included(
        self,
        perfect_agreement_scores: list[Score],
        human_scores_for_agreement: list[HumanScore],
    ) -> None:
        alpha = krippendorff_alpha(
            perfect_agreement_scores, human_scores_for_agreement
        )
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_single_rater_returns_nan(self) -> None:
        """α is undefined for a single rater — should return NaN, not raise."""
        from swe_judge.tasks import Score as S

        scores = [
            S(run_id="R", task_id=f"t{i}", judge_model="solo",
              dimension="correctness", value=v,
              rationale="x", anchor_matched=f"{v}")
            for i, v in enumerate([3, 4, 5])
        ]
        result = krippendorff_alpha(scores)
        assert math.isnan(result)

    def test_zero_variance_returns_nan(self) -> None:
        """α is undefined when every score is the same value — should return NaN."""
        from swe_judge.tasks import Score as S

        scores = [
            S(run_id="R", task_id=f"t{i}", judge_model=j,
              dimension="correctness", value=3,
              rationale="x", anchor_matched="3")
            for i in range(3)
            for j in ["A", "B"]
        ]
        result = krippendorff_alpha(scores)
        assert math.isnan(result)

    def test_empty_input_returns_nan(self) -> None:
        result = krippendorff_alpha([])
        assert math.isnan(result)


class TestSummary:
    def test_summary_keys(
        self,
        perfect_agreement_scores: list[Score],
        human_scores_for_agreement: list[HumanScore],
    ) -> None:
        s = summary(perfect_agreement_scores, human_scores_for_agreement)
        assert "n_scores" in s
        assert "mean_inter_judge_kappa" in s
        assert "mean_judge_vs_human_kappa" in s
        assert "krippendorff_alpha_with_human" in s
        assert s["n_scores"] == len(perfect_agreement_scores)

    def test_summary_without_human(self, perfect_agreement_scores: list[Score]) -> None:
        s = summary(perfect_agreement_scores)
        assert "krippendorff_alpha" in s
        assert "mean_judge_vs_human_kappa" not in s
