"""
Statistical comparison between two ablation configs on a shared metric.

The ablation runner's console summary reports mean metric values per
config, which invites reading noise as signal on an 8-query sample
corpus: "hybrid got 0.62 recall vs bm25's 0.58" looks like a finding but
could easily be one query flipping. This module answers the actual
question -- "is that difference distinguishable from chance, given how
much these per-query scores vary?" -- with two standard, dependency-free
methods:

- Bootstrap confidence interval: resample the per-query metric values
  with replacement many times, recompute the mean each time, and read
  off the 2.5th/97.5th percentiles. Answers "how much would this mean
  move if I'd sampled slightly different queries?".

- Paired permutation test: for two configs evaluated on the *same*
  queries (the common case -- every config in an ablation sweep runs
  against the same eval set), randomly flip the sign of each paired
  difference many times and see how often the resulting mean difference
  is as extreme as the one actually observed. Nonparametric, no
  normality assumption, appropriate for small samples -- important here
  since eval sets are typically 8-50 labeled queries, not thousands.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Returns (mean, lower_bound, upper_bound) for the given confidence
    level, via the percentile bootstrap. `values` should be per-query
    scores for a single config on a single metric (e.g. every row's
    recall@5 for one chunk_strategy/retriever/top_k combination)."""
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = 1 - ci
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    mean = sum(values) / n
    return mean, means[lo_idx], means[hi_idx]


def paired_permutation_test(
    values_a: list[float], values_b: list[float], n_permutations: int = 10000, seed: int = 0
) -> float:
    """Two-sided p-value for whether mean(values_a) - mean(values_b)
    differs from zero more than chance, given paired (same query, two
    configs) observations. `values_a` and `values_b` must be the same
    length and index-aligned (values_a[i] and values_b[i] come from the
    same query_id).
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"paired_permutation_test requires equal-length paired samples, "
            f"got {len(values_a)} vs {len(values_b)}"
        )
    if not values_a:
        return 1.0

    diffs = [a - b for a, b in zip(values_a, values_b)]
    observed = sum(diffs) / len(diffs)
    if observed == 0.0 and all(d == 0.0 for d in diffs):
        return 1.0

    rng = random.Random(seed)
    n = len(diffs)
    as_extreme = 0
    for _ in range(n_permutations):
        signed = [d if rng.random() < 0.5 else -d for d in diffs]
        permuted_mean = sum(signed) / n
        if abs(permuted_mean) >= abs(observed) - 1e-12:
            as_extreme += 1
    return as_extreme / n_permutations


@dataclass
class ComparisonResult:
    metric: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    p_value: float

    def summary(self) -> str:
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"{self.metric}: a={self.mean_a:.3f} {self.ci_a}  b={self.mean_b:.3f} {self.ci_b}  "
            f"diff={self.diff:+.3f}  p={self.p_value:.4f} ({sig} at alpha=0.05, n={self.n})"
        )


def compare_configs(
    rows_a: list[dict], rows_b: list[dict], metric: str,
    query_id_key: str = "query_id", n_permutations: int = 10000, seed: int = 0,
) -> ComparisonResult:
    """Compares two configs' results (as produced by eval.ablation.run_ablation,
    already filtered down to the two configs of interest) on one metric
    column. Rows are matched by query_id_key so the permutation test is
    correctly paired even if the rows arrive in different order.
    """
    by_id_a = {r[query_id_key]: r[metric] for r in rows_a}
    by_id_b = {r[query_id_key]: r[metric] for r in rows_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))
    if not shared_ids:
        raise ValueError(
            f"No shared {query_id_key} values between the two row sets -- "
            f"can't run a paired comparison."
        )
    values_a = [by_id_a[qid] for qid in shared_ids]
    values_b = [by_id_b[qid] for qid in shared_ids]

    mean_a, lo_a, hi_a = bootstrap_ci(values_a, seed=seed)
    mean_b, lo_b, hi_b = bootstrap_ci(values_b, seed=seed)
    p = paired_permutation_test(values_a, values_b, n_permutations=n_permutations, seed=seed)

    return ComparisonResult(
        metric=metric,
        n=len(shared_ids),
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_a - mean_b,
        ci_a=(round(lo_a, 4), round(hi_a, 4)),
        ci_b=(round(lo_b, 4), round(hi_b, 4)),
        p_value=p,
    )
