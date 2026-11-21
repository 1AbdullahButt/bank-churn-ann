"""Tests for predict_cli.py. Each test runs the script as a real command."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "predict_cli.py"

VALID = {
    "--credit-score": "650",
    "--geography": "France",
    "--gender": "Male",
    "--age": "38",
    "--tenure": "5",
    "--balance": "76000",
    "--num-of-products": "1",
    "--has-cr-card": "1",
    "--is-active-member": "1",
    "--estimated-salary": "100000",
}


def run_cli(*extra, **overrides):
    fields = {**VALID, **overrides}
    args = [x for pair in fields.items() for x in pair] + list(extra)
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=300)


def test_valid_customer_gives_probability_between_0_and_1():
    result = run_cli()
    assert result.returncode == 0, result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("Churn probability"))
    probability = float(line.split("(")[1].rstrip(")"))
    assert 0.0 <= probability <= 1.0
    assert "Verdict: likely to" in result.stdout


def test_unknown_geography_fails_cleanly():
    result = run_cli(**{"--geography": "Italy"})
    assert result.returncode != 0
    assert "unknown Geography 'Italy'" in result.stderr
    assert "France, Germany, Spain" in result.stderr
    assert "Traceback" not in result.stderr


def test_non_numeric_value_fails_cleanly():
    result = run_cli(**{"--age": "forty"})
    assert result.returncode != 0
    assert "--age" in result.stderr and "not a whole number" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_model_file_fails_cleanly(tmp_path):
    missing = tmp_path / "nope.keras"
    result = run_cli("--model", str(missing))
    assert result.returncode != 0
    assert "model file not found" in result.stderr
    assert "Traceback" not in result.stderr
