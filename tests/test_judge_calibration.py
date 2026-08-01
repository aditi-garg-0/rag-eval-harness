from eval.judge_calibration import (
    calibration_report, CalibrationExample, load_human_labels,
)


def test_perfect_agreement_gives_kappa_one():
    judge_scores = {"q1": 5, "q2": 4, "q3": 3, "q4": 2, "q5": 1}
    labels = [
        CalibrationExample("q1", "faithfulness", 5),
        CalibrationExample("q2", "faithfulness", 4),
        CalibrationExample("q3", "faithfulness", 3),
        CalibrationExample("q4", "faithfulness", 2),
        CalibrationExample("q5", "faithfulness", 1),
    ]
    report = calibration_report(judge_scores, labels, metric="faithfulness")
    assert report.n == 5
    assert abs(report.weighted_kappa - 1.0) < 1e-9
    assert report.mean_absolute_error == 0.0
    assert report.exact_match_rate == 1.0


def test_off_by_one_scores_report_within_one_but_not_exact():
    judge_scores = {"q1": 4, "q2": 3}
    labels = [
        CalibrationExample("q1", "faithfulness", 5),
        CalibrationExample("q2", "faithfulness", 4),
    ]
    report = calibration_report(judge_scores, labels, metric="faithfulness")
    assert report.exact_match_rate == 0.0
    assert report.within_one_point_rate == 1.0
    assert report.mean_absolute_error == 1.0


def test_only_matching_metric_and_query_id_are_used():
    judge_scores = {"q1": 5, "q2": 5}
    labels = [
        CalibrationExample("q1", "faithfulness", 5),
        CalibrationExample("q2", "relevance", 1),   # different metric, excluded
        CalibrationExample("q3", "faithfulness", 1),  # not in judge_scores, excluded
    ]
    report = calibration_report(judge_scores, labels, metric="faithfulness")
    assert report.n == 1


def test_no_matching_examples_returns_zeroed_report():
    report = calibration_report({}, [], metric="faithfulness")
    assert report.n == 0
    assert report.weighted_kappa == 0.0


def test_load_human_labels_roundtrip(tmp_path):
    import json
    path = tmp_path / "labels.json"
    payload = [
        {"query_id": "q1", "metric": "faithfulness", "human_score": 4, "notes": "solid"},
        {"query_id": "q2", "metric": "relevance", "human_score": 2},
    ]
    path.write_text(json.dumps(payload))
    labels = load_human_labels(path)
    assert len(labels) == 2
    assert labels[0].query_id == "q1"
    assert labels[0].human_score == 4
    assert labels[1].notes == ""
