"""Predict churn for MANY customers from a CSV file.

Example:
    python batch_predict.py customers.csv -o predictions.csv

The input CSV needs these columns (any order, extra columns are kept):
    CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
    HasCrCard, IsActiveMember, EstimatedSalary

The output CSV has every good input row with its original columns plus a
churn_probability column. It uses the same model.keras, preprocessor.pkl and
column order as app.py. A row with a bad value is skipped with a warning on
stderr (its line number and the reason); the other rows are still predicted.

Exit codes: 0 = done (maybe with skipped rows), 1 = no usable rows or a
missing/unloadable model or preprocessor, 2 = bad arguments or a missing column.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Same column order as the one-row table in app.py
COLUMNS = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
           "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
# Same ranges as the inputs in app.py: column -> (type, min, max)
NUMBERS = {
    "CreditScore": (int, 300, 900),
    "Age": (int, 18, 92),
    "Tenure": (int, 0, 10),
    "Balance": (float, 0.0, 260000.0),
    "NumOfProducts": (int, 1, 4),
    "HasCrCard": (int, 0, 1),
    "IsActiveMember": (int, 0, 1),
    "EstimatedSalary": (float, 0.0, 200000.0),
}
CATEGORIES = ["Geography", "Gender"]
OUTPUT_COLUMN = "churn_probability"


class BatchError(Exception):
    """An error with a message meant for the user (no traceback)."""

    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def parse_number(text, kind):
    """'650' -> 650, '650.0' -> 650 for int columns; raises ValueError if not a number."""
    value = float(text)
    if value != value:  # NaN
        raise ValueError
    if kind is int:
        if value != int(value):
            raise ValueError
        return int(value)
    return value


def known_categories(preprocessor):
    """{'Geography': {'france': 'France', ...}, 'Gender': {...}} from the fitted encoder.

    The encoder ignores unknown values (all-zero encoding), which would give a
    quiet wrong prediction, so rows are checked against these lists first.
    """
    for _, transformer, cols in getattr(preprocessor, "transformers_", []):
        cats = getattr(transformer, "categories_", None)
        if cats is not None and set(CATEGORIES) <= set(cols):
            return {c: {str(v).lower(): str(v) for v in cats[list(cols).index(c)]}
                    for c in CATEGORIES}
    return None  # unknown preprocessor layout: let transform() decide


def check_row(row, known):
    """Return (clean values dict, None) or (None, reason)."""
    clean = {}
    for col in COLUMNS:
        text = (row.get(col) or "").strip()
        if text == "":
            return None, f"{col} is empty"
        if col in NUMBERS:
            kind, lo, hi = NUMBERS[col]
            try:
                value = parse_number(text, kind)
            except ValueError:
                word = "a whole number" if kind is int else "a number"
                return None, f"{col}={text!r} is not {word}"
            if not lo <= value <= hi:
                return None, f"{col}={text} is out of range ({lo} to {hi})"
            clean[col] = value
        else:
            if known is not None:
                match = known[col].get(text.lower())
                if match is None:
                    allowed = ", ".join(sorted(known[col].values()))
                    return None, f"{col}={text!r} is not one the model knows ({allowed})"
                text = match
            clean[col] = text
    return clean, None


def load_model_files(model_path, preprocessor_path):
    for path in (model_path, preprocessor_path):
        if not Path(path).is_file():
            raise BatchError(f"error: file not found: {path}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import joblib
    from tensorflow import keras
    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as exc:  # noqa: BLE001 - any load failure is a user-facing error
        raise BatchError(f"error: could not load preprocessor {preprocessor_path}: {exc}")
    try:
        model = keras.models.load_model(model_path)
    except Exception as exc:  # noqa: BLE001
        raise BatchError(f"error: could not load model {model_path}: {exc}")
    return model, preprocessor


def run(input_path, output_path, model_path, preprocessor_path):
    """Predict for every good row. Returns (rows written, rows skipped)."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise BatchError(f"error: input file not found: {input_path}", code=2)
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        reader.fieldnames = header
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise BatchError(f"error: {input_path} is missing column(s): {', '.join(missing)}",
                             code=2)
        rows = list(reader)

    model, preprocessor = load_model_files(model_path, preprocessor_path)
    known = known_categories(preprocessor)

    good, clean_values, skipped = [], [], 0
    for line_no, row in enumerate(rows, start=2):  # line 1 is the header
        if None in row:  # more fields than the header has
            reason = "has more values than the header"
            clean = None
        else:
            clean, reason = check_row(row, known)
        if clean is None:
            print(f"warning: line {line_no}: {reason}; row skipped", file=sys.stderr)
            skipped += 1
            continue
        good.append(row)
        clean_values.append(clean)

    if not good:
        raise BatchError(f"error: no usable rows in {input_path} ({skipped} skipped)")

    import pandas as pd
    table = pd.DataFrame(clean_values, columns=COLUMNS)  # same columns/order as app.py
    X_new = preprocessor.transform(table)
    probabilities = model.predict(X_new, verbose=0)[:, 0]

    out_header = header + ([OUTPUT_COLUMN] if OUTPUT_COLUMN not in header else [])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        for row, p in zip(good, probabilities):
            writer.writerow({**row, OUTPUT_COLUMN: f"{float(p):.4f}"})
    return len(good), skipped


def build_parser():
    p = argparse.ArgumentParser(
        description="Predict churn for every customer in a CSV file and write the "
                    "results to a new CSV with a churn_probability column.")
    p.add_argument("input", help="input CSV with the customer columns")
    p.add_argument("-o", "--output", default="predictions.csv",
                   help="output CSV (default: predictions.csv)")
    p.add_argument("--model", default=str(HERE / "model.keras"),
                   help="path to model.keras (default: next to this script)")
    p.add_argument("--preprocessor", default=str(HERE / "preprocessor.pkl"),
                   help="path to preprocessor.pkl (default: next to this script)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        written, skipped = run(args.input, args.output, args.model, args.preprocessor)
    except BatchError as exc:
        print(exc, file=sys.stderr)
        return exc.code
    print(f"Wrote {written} prediction(s) to {args.output}"
          + (f" ({skipped} row(s) skipped, see warnings)" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
