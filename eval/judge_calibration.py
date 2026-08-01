"""
Judge calibration: measures how much to trust the LLM-as-judge scores
from eval/judge.py, against a small set of human-labeled examples.

The README states this project's known confound up front: using the
same small local model as both generator and judge means the judge may
be systematically lenient (or harsh) toward its own outputs, in ways a
GPT-4-class judge in papers like RAGAS/ARES wouldn't be. This module is
how you go from "stated limitation" to "measured limitation": load a
handful of human 1-5 scores for the same (question, answer, context)
triples the judge scored, and report:

- Linear-weighted Cohen's kappa: agreement beyond chance, appropriate
  for ordinal 1-5 scales (an unweighted kappa would treat a judge/human
  gap of 5-vs-4 the same as 5-vs-1, which isn't the right notion of
  "disagreement" for a rating scale).
- Mean absolute error: average |judge_score - human_score|, an easier
  to interpret companion number ("off by about N points on average").
- Exact-match rate and within-1-point agreement rate, since kappa alone
  can be hard to build intuition for.

This needs a real human-labeled file to be useful (see
`load_human_labels` for the expected format) -- it does not replace
having a human actually look at outputs, it just makes "how much do we
trust the judge" a reportable, falsifiable number instead of a caveat
in prose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrationExample:
    query_id: str
    metric: str  # e.g. "faithfulness" or "relevance"
    human_score: int  # 1-5, assigned by a person reading the same triple
    notes: str = ""


def load_human_labels(path: str | Path) -> list[CalibrationExample]:
    """Loads human labels from a JSON file: a list of objects with
    query_id, metric, human_score, and optional notes. See
    data/sample_corpus/sample_judge_calibration.json for the format.
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    return [
        CalibrationExample(
            query_id=item["query_id"],
            metric=item["metric"],
            human_score=int(item["human_score"]),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


def _weighted_kappa(pairs: list[tuple[int, int]], n_categories: int = 5) -> float:
    """Linear-weighted Cohen's kappa over an ordinal 1..n_categories
    scale. Implemented from the standard definition (weighted observed
    vs. expected disagreement) rather than pulled in from a dependency,
    since this is the one piece of the harness that would otherwise need
    scipy/sklearn just for one formula.
    """
    if not pairs:
        return 0.0

    n = n_categories
    # Confusion matrix over categories 1..n (index 0..n-1).
    counts = [[0] * n for _ in range(n)]
    for judge_s, human_s in pairs:
        j = max(1, min(n, judge_s)) - 1
        h = max(1, min(n, human_s)) - 1
        counts[j][h] += 1

    total = len(pairs)
    row_marginals = [sum(row) / total for row in counts]
    col_marginals = [sum(counts[r][c] for r in range(n)) / total for c in range(n)]

    weights = [[1 - (abs(i - j) / (n - 1)) ** 2 for j in range(n)] for i in range(n)]

    observed = sum(
        weights[i][j] * (counts[i][j] / total) for i in range(n) for j in range(n)
    )
    expected = sum(
        weights[i][j] * row_marginals[i] * col_marginals[j] for i in range(n) for j in range(n)
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


@dataclass
class CalibrationReport:
    metric: str
    n: int
    weighted_kappa: float
    mean_absolute_error: float
    exact_match_rate: float
    within_one_point_rate: float

    def summary(self) -> str:
        return (
            f"{self.metric} (n={self.n}): weighted kappa={self.weighted_kappa:.3f}  "
            f"MAE={self.mean_absolute_error:.2f}  exact match={self.exact_match_rate:.1%}  "
            f"within 1 point={self.within_one_point_rate:.1%}"
        )


def calibration_report(
    judge_scores: dict[str, int],
    human_labels: list[CalibrationExample],
    metric: str,
) -> CalibrationReport:
    """Compares judge_scores (query_id -> judge's 1-5 score, for one
    metric) against human_labels filtered to that metric. Only
    query_ids present in both are used.
    """
    relevant_labels = [h for h in human_labels if h.metric == metric]
    pairs = [
        (judge_scores[h.query_id], h.human_score)
        for h in relevant_labels
        if h.query_id in judge_scores
    ]
    if not pairs:
        return CalibrationReport(
            metric=metric, n=0, weighted_kappa=0.0,
            mean_absolute_error=0.0, exact_match_rate=0.0, within_one_point_rate=0.0,
        )

    n = len(pairs)
    mae = sum(abs(j - h) for j, h in pairs) / n
    exact = sum(1 for j, h in pairs if j == h) / n
    within_one = sum(1 for j, h in pairs if abs(j - h) <= 1) / n
    kappa = _weighted_kappa(pairs)

    return CalibrationReport(
        metric=metric, n=n, weighted_kappa=kappa,
        mean_absolute_error=mae, exact_match_rate=exact, within_one_point_rate=within_one,
    )
