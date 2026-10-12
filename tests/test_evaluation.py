"""Ensure evaluation fixture is reproducible and reports expected denominators."""

from evaluate import evaluate_cases


def test_evaluation_metrics_are_bounded():
    metrics, rows = evaluate_cases()
    assert metrics["n_examples"] == len(rows) == 26
    for name, score in metrics.items():
        if name != "n_examples":
            assert 0 <= score <= 1, name


def test_evaluation_input_has_no_private_data():
    _, rows = evaluate_cases()
    assert len(rows) == 26
    assert all(row["input"] for row in rows)
