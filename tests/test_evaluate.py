from pathlib import Path

import numpy as np
import pytest

from src.evaluate import accuracy, confusion, recall, report, write_results

Y_TRUE = np.array([0, 1, 0, 1, 1, 0])
Y_PERFECT = np.array([0, 1, 0, 1, 1, 0])
Y_WRONG = np.array([1, 0, 1, 0, 0, 1])
Y_MIXED = np.array([0, 1, 1, 1, 0, 0])


def test_accuracy_perfect() -> None:
    assert accuracy(Y_TRUE, Y_PERFECT) == 1.0


def test_accuracy_all_wrong() -> None:
    assert accuracy(Y_TRUE, Y_WRONG) == 0.0


def test_recall_returns_float() -> None:
    assert isinstance(recall(Y_TRUE, Y_PERFECT), float)


def test_confusion_shape() -> None:
    assert confusion(Y_TRUE, Y_MIXED).shape == (2, 2)


def test_report_contains_name() -> None:
    r = report("My Model", Y_TRUE, Y_MIXED)
    assert "My Model" in r


def test_write_results_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "results.md"
    write_results(["## Model A\n\nsome content"], out)
    assert out.exists()
    assert "Model A" in out.read_text()
