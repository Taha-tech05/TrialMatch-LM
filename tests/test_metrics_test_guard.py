"""The test split is read by the metrics scripts only with --split test --confirm-final."""
import subprocess
import sys
from pathlib import Path

import pytest

from trialmatch.evaluation.reporting import build_metrics, comparison_markdown
from trialmatch.evaluation.val_data import check_metrics_split, load_gold

ROOT = Path(__file__).resolve().parents[1]


def test_guard_logic():
    assert check_metrics_split("val", False) is False
    assert check_metrics_split("test", True) is True
    with pytest.raises(PermissionError, match="confirm-final"):
        check_metrics_split("test", False)
    with pytest.raises(ValueError):
        check_metrics_split("val", True)            # the flag alone must not mean anything on val
    with pytest.raises(ValueError):
        check_metrics_split("train", False)


def test_load_gold_still_refuses_test_by_default():
    with pytest.raises(PermissionError):
        load_gold("test")
    with pytest.raises(PermissionError):
        load_gold("test", allow_test=False)


@pytest.mark.parametrize("script", ["compute_metrics.py", "export_errors.py"])
def test_scripts_refuse_test_without_flag(script):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--split", "test"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "confirm-final" in r.stderr


def test_script_refuses_flag_alone_on_val():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compute_metrics.py"), "--confirm-final"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0


def test_split_aware_report_labels():
    rec = [{"annotation_id": 1, "criterion_type": "inclusion", "gold_label": "included", "status": "ok",
            "pred_label": "included", "gold_evidence": [0], "pred_evidence": [0]},
           {"annotation_id": 2, "criterion_type": "exclusion", "gold_label": "not excluded", "status": "ok",
            "pred_label": "not excluded", "gold_evidence": [1], "pred_evidence": [1]}]
    m = {"x": build_metrics("x", rec, split="test")}
    assert m["x"]["split"] == "test"
    md = comparison_markdown(m, split="test", preamble=["PRE"])
    assert md.startswith("# Test comparison") and "PRE" in md and "Test split" in md and "Validation split only" not in md
    assert comparison_markdown(m).startswith("# Validation comparison")
