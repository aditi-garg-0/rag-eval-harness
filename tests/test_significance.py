import pytest

from eval.significance import bootstrap_ci, paired_permutation_test, compare_configs


def test_bootstrap_ci_mean_is_sample_mean():
    values = [0.2, 0.4, 0.6, 0.8]
    mean, lo, hi = bootstrap_ci(values, n_resamples=1000, seed=0)
    assert abs(mean - 0.5) < 1e-9
    assert lo <= mean <= hi


def test_bootstrap_ci_single_value():
    mean, lo, hi = bootstrap_ci([0.7])
    assert mean == lo == hi == 0.7


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_tighter_with_less_variance():
    tight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.51, 0.49]
    wide = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
    _, lo_t, hi_t = bootstrap_ci(tight, n_resamples=2000, seed=1)
    _, lo_w, hi_w = bootstrap_ci(wide, n_resamples=2000, seed=1)
    assert (hi_t - lo_t) < (hi_w - lo_w)


def test_paired_permutation_identical_values_p_is_one():
    p = paired_permutation_test([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert p == 1.0


def test_paired_permutation_clearly_different_is_significant():
    a = [1.0] * 8
    b = [0.0] * 8
    p = paired_permutation_test(a, b, n_permutations=2000, seed=0)
    assert p < 0.05


def test_paired_permutation_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([1, 2], [1, 2, 3])


def test_compare_configs_matches_on_query_id_and_ignores_extras():
    rows_a = [
        {"query_id": "q1", "recall@3": 1.0},
        {"query_id": "q2", "recall@3": 1.0},
        {"query_id": "q3", "recall@3": 1.0},  # no counterpart in rows_b
    ]
    rows_b = [
        {"query_id": "q1", "recall@3": 0.0},
        {"query_id": "q2", "recall@3": 0.0},
    ]
    result = compare_configs(rows_a, rows_b, metric="recall@3", n_permutations=500)
    assert result.n == 2  # only q1, q2 are shared
    assert result.mean_a == 1.0
    assert result.mean_b == 0.0
    assert result.diff == 1.0


def test_compare_configs_no_shared_ids_raises():
    rows_a = [{"query_id": "q1", "recall@3": 1.0}]
    rows_b = [{"query_id": "q2", "recall@3": 0.0}]
    with pytest.raises(ValueError):
        compare_configs(rows_a, rows_b, metric="recall@3")
