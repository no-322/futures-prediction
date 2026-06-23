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


# --- no-flat slice (keep mask) -------------------------------------------------

# Two flat rows (indices 1, 4) marked False in the keep mask.
KEEP = np.array([True, False, True, True, False, True])


def test_keep_mask_excludes_flat() -> None:
    keep = KEEP
    yt, yp = Y_TRUE[keep], Y_MIXED[keep]
    assert accuracy(Y_TRUE, Y_MIXED, keep) == pytest.approx(accuracy(yt, yp))
    assert recall(Y_TRUE, Y_MIXED, keep) == pytest.approx(recall(yt, yp))
    cm = confusion(Y_TRUE, Y_MIXED, keep)
    assert int(cm.sum()) == int(keep.sum())          # only kept rows counted


def test_include_flat_uses_full() -> None:
    full = accuracy(Y_TRUE, Y_MIXED, include_flat=True)
    assert full == pytest.approx(accuracy(Y_TRUE, Y_MIXED))           # ignores keep
    assert accuracy(Y_TRUE, Y_MIXED, KEEP, include_flat=True) == pytest.approx(full)


def test_keep_none_back_compat() -> None:
    # Omitting keep reproduces the legacy full-set behaviour.
    assert accuracy(Y_TRUE, Y_MIXED) == pytest.approx(
        float((Y_TRUE == Y_MIXED).mean())
    )
    assert confusion(Y_TRUE, Y_MIXED).shape == (2, 2)


def test_report_notes_slice() -> None:
    r_noflat = report("M", Y_TRUE, Y_MIXED, KEEP)
    assert "non-flat" in r_noflat
    assert f"{int(KEEP.sum()):,}" in r_noflat
    r_full = report("M", Y_TRUE, Y_MIXED, KEEP, include_flat=True)
    assert "full set" in r_full
