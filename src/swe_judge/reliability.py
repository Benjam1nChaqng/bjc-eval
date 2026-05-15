"""Inter-rater reliability calculations.

Implements Cohen's κ (kappa) for pairwise agreement and Krippendorff's α
for multi-rater agreement on ordinal data. These are the headline metrics
the README reports.

Conventions used here (consistent with sklearn / Landis & Koch 1977):
    κ < 0.0   : poor (worse than chance)
    0.0–0.2  : slight
    0.2–0.4  : fair
    0.4–0.6  : moderate
    0.6–0.8  : substantial   ← target for v0.1 inter-judge agreement
    0.8–1.0  : almost perfect
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

from swe_judge.tasks import Dimension, HumanScore, Score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_scores_by_judge(
    scores: Iterable[Score],
) -> dict[str, dict[tuple[str, Dimension], int]]:
    """Group scores by judge, indexed by (task_id, dimension) → value."""
    indexed: dict[str, dict[tuple[str, Dimension], int]] = defaultdict(dict)
    for s in scores:
        indexed[s.judge_model][(s.task_id, s.dimension)] = s.value
    return dict(indexed)


def _index_human_scores(
    human_scores: Iterable[HumanScore],
) -> dict[tuple[str, Dimension], int]:
    """Index human scores by (task_id, dimension) → value."""
    return {(h.task_id, h.dimension): h.value for h in human_scores}


def _aligned_pair(
    a: dict[tuple[str, Dimension], int],
    b: dict[tuple[str, Dimension], int],
) -> tuple[list[int], list[int]]:
    """Return two parallel lists of scores for keys present in BOTH a and b."""
    common = sorted(a.keys() & b.keys())
    return [a[k] for k in common], [b[k] for k in common]


# ---------------------------------------------------------------------------
# Cohen's kappa (pairwise)
# ---------------------------------------------------------------------------


def cohen_kappa_pair(
    scores_a: Iterable[Score],
    scores_b: Iterable[Score],
) -> float:
    """Cohen's κ between two raters (judges) on their shared (task, dim) pairs.

    The scores can come from any single judge — caller decides whose scores
    are A and whose are B. Uses linear weights since the scale is ordinal.

    Raises:
        ValueError: if there are fewer than 2 common (task, dim) entries.
    """
    a_idx = _index_scores_by_judge(scores_a)
    b_idx = _index_scores_by_judge(scores_b)

    # Caller is expected to pass scores from a single judge each; flatten.
    a_flat: dict[tuple[str, Dimension], int] = {}
    for d in a_idx.values():
        a_flat.update(d)
    b_flat: dict[tuple[str, Dimension], int] = {}
    for d in b_idx.values():
        b_flat.update(d)

    aligned_a, aligned_b = _aligned_pair(a_flat, b_flat)
    if len(aligned_a) < 2:
        raise ValueError(
            f"Need at least 2 common (task_id, dimension) entries; got {len(aligned_a)}"
        )

    return float(cohen_kappa_score(aligned_a, aligned_b, weights="linear"))


def inter_judge_agreement(scores: Iterable[Score]) -> dict[tuple[str, str], float]:
    """Compute pairwise Cohen's κ between every distinct pair of judges.

    Returns dict keyed by sorted (judge_a, judge_b) tuples.
    """
    by_judge = _index_scores_by_judge(scores)
    judges = sorted(by_judge.keys())
    out: dict[tuple[str, str], float] = {}
    for ja, jb in combinations(judges, 2):
        aligned_a, aligned_b = _aligned_pair(by_judge[ja], by_judge[jb])
        if len(aligned_a) < 2:
            continue
        out[(ja, jb)] = float(cohen_kappa_score(aligned_a, aligned_b, weights="linear"))
    return out


def mean_inter_judge_kappa(scores: Iterable[Score]) -> float:
    """The headline number for the README: mean pairwise Cohen's κ."""
    pairs = inter_judge_agreement(scores)
    if not pairs:
        return float("nan")
    return sum(pairs.values()) / len(pairs)


# ---------------------------------------------------------------------------
# Judge vs. human ground truth
# ---------------------------------------------------------------------------


def judge_vs_human(
    scores: Iterable[Score],
    human_scores: Iterable[HumanScore],
) -> dict[str, float]:
    """Cohen's κ between each judge and the human scorer.

    Returns dict keyed by judge_model. Only computes for judges with ≥2
    overlapping (task, dim) entries with the human ground truth.
    """
    by_judge = _index_scores_by_judge(scores)
    human_idx = _index_human_scores(human_scores)
    out: dict[str, float] = {}
    for judge, judge_idx in by_judge.items():
        aligned_j, aligned_h = _aligned_pair(judge_idx, human_idx)
        if len(aligned_j) < 2:
            continue
        out[judge] = float(cohen_kappa_score(aligned_j, aligned_h, weights="linear"))
    return out


def mean_judge_vs_human(
    scores: Iterable[Score],
    human_scores: Iterable[HumanScore],
) -> float:
    """The second headline number: mean κ across judges vs. human ground truth."""
    per_judge = judge_vs_human(scores, human_scores)
    if not per_judge:
        return float("nan")
    return sum(per_judge.values()) / len(per_judge)


# ---------------------------------------------------------------------------
# Krippendorff's alpha (treats all raters as one wide table)
# ---------------------------------------------------------------------------


def krippendorff_alpha(
    scores: Iterable[Score],
    human_scores: Iterable[HumanScore] | None = None,
) -> float:
    """Krippendorff's α on ordinal data across all raters.

    If human_scores is provided, the human is included as an additional rater.
    Uses simpledorff which supports ordinal level of measurement.

    Returns NaN when α is mathematically undefined — most commonly when
    there is only one rater or when every score has the same value (zero
    variance, so expected disagreement is 0 → 0/0). This matches the
    convention of returning NaN for undefined statistics rather than
    raising at compute time.
    """
    import pandas as pd  # local import; pandas only needed here
    import simpledorff  # type: ignore[import-untyped]

    rows: list[dict[str, str | int]] = []
    for s in scores:
        rows.append(
            {
                "unit": f"{s.task_id}::{s.dimension}",
                "rater": s.judge_model,
                "value": s.value,
            }
        )
    if human_scores:
        for h in human_scores:
            rows.append(
                {
                    "unit": f"{h.task_id}::{h.dimension}",
                    "rater": f"human::{h.scorer}",
                    "value": h.value,
                }
            )

    if not rows:
        return float("nan")

    df = pd.DataFrame(rows)

    # Short-circuit: α is undefined with a single rater or zero-variance values.
    if df["rater"].nunique() < 2 or df["value"].nunique() < 2:
        return float("nan")

    try:
        return float(
            simpledorff.calculate_krippendorffs_alpha_for_df(
                df,
                experiment_col="unit",
                annotator_col="rater",
                class_col="value",
            )
        )
    except ZeroDivisionError:
        # Defensive: even with variance in the input, certain pathological
        # distributions can still produce zero expected disagreement.
        return float("nan")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summary(
    scores: list[Score],
    human_scores: list[HumanScore] | None = None,
) -> dict[str, object]:
    """Compute all headline reliability metrics in one shot.

    This is what the CLI prints at the end of a run.
    """
    pairs = inter_judge_agreement(scores)
    summary_dict: dict[str, object] = {
        "n_scores": len(scores),
        "inter_judge_pairwise": {f"{a} <-> {b}": k for (a, b), k in pairs.items()},
        "mean_inter_judge_kappa": mean_inter_judge_kappa(scores),
    }
    if human_scores:
        summary_dict["judge_vs_human"] = judge_vs_human(scores, human_scores)
        summary_dict["mean_judge_vs_human_kappa"] = mean_judge_vs_human(scores, human_scores)
        summary_dict["krippendorff_alpha_with_human"] = krippendorff_alpha(
            scores, human_scores
        )
    else:
        summary_dict["krippendorff_alpha"] = krippendorff_alpha(scores)
    return summary_dict
