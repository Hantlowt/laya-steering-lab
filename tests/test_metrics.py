import pytest

from laya_steering.metrics import classification_metrics, option_order_instability, paired_summary


def test_metrics_known_values():
    result = classification_metrics(
        ["A", "A", "B", "B"],
        [{"A": 0.9, "B": 0.1}, {"A": 0.4, "B": 0.6}, {"A": 0.2, "B": 0.8}, {"A": 0.7, "B": 0.3}],
        ["A", "B"],
    )
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["macro_f1"] == 0.5
    assert result["log_loss"] > 0
    assert 0 <= result["ece"] <= 1


def test_option_order_instability():
    assert option_order_instability([["A", "B", "A"], ["A", "A", "A"]]) == pytest.approx(1 / 3)


def test_paired_summary_reports_wins_ties_losses():
    result = paired_summary({"a": 0.8, "b": 0.4, "c": 0.5}, {"a": 0.7, "b": 0.5, "c": 0.5})
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
    assert result["paired_difference"] == pytest.approx(0)
