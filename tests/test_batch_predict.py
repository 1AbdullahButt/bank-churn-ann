"""Tests for batch_predict.py. Run with: python -m pytest"""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import batch_predict  # noqa: E402

HEADER = ("CustomerId,CreditScore,Geography,Gender,Age,Tenure,Balance,"
          "NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary")
GOOD_1 = "101,650,France,Male,38,5,76000,1,1,1,100000"
GOOD_2 = "102,520,Germany,Female,55,2,120000.50,3,0,0,45000"


def write_csv(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_valid_rows_get_a_probability(tmp_path):
    src = write_csv(tmp_path / "in.csv", [HEADER, GOOD_1, GOOD_2])
    out = tmp_path / "out.csv"
    assert batch_predict.main([str(src), "-o", str(out)]) == 0
    rows = read_csv(out)
    assert [r["CustomerId"] for r in rows] == ["101", "102"]  # extra column kept, order kept
    for r in rows:
        assert 0.0 <= float(r["churn_probability"]) <= 1.0


def test_bad_rows_are_skipped_with_a_warning(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", [
        HEADER,
        GOOD_1,
        "103,abc,France,Male,38,5,76000,1,1,1,100000",      # non-numeric
        "104,650,Italy,Male,38,5,76000,1,1,1,100000",       # unknown Geography
        "105,650,Spain,Female,150,5,76000,1,1,1,100000",    # Age out of range
        GOOD_2,
    ])
    out = tmp_path / "out.csv"
    assert batch_predict.main([str(src), "-o", str(out)]) == 0
    assert [r["CustomerId"] for r in read_csv(out)] == ["101", "102"]
    err = capsys.readouterr().err
    assert "line 3" in err and "CreditScore" in err
    assert "line 4" in err and "Geography" in err
    assert "line 5" in err and "Age" in err
    assert "Traceback" not in err


def test_missing_column_fails_cleanly(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", [
        HEADER.replace(",Gender", ""), "101,650,France,38,5,76000,1,1,1,100000"])
    out = tmp_path / "out.csv"
    assert batch_predict.main([str(src), "-o", str(out)]) == 2
    assert "missing column(s): Gender" in capsys.readouterr().err
    assert not out.exists()


def test_missing_model_file_fails_cleanly(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", [HEADER, GOOD_1])
    out = tmp_path / "out.csv"
    code = batch_predict.main([str(src), "-o", str(out),
                               "--model", str(tmp_path / "nope.keras")])
    assert code == 1
    err = capsys.readouterr().err
    assert "file not found" in err and "nope.keras" in err
    assert not out.exists()
