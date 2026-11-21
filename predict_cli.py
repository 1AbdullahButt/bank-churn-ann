"""Predict churn for ONE customer from the command line.

Example:
    python predict_cli.py --credit-score 650 --geography France --gender Male \
        --age 38 --tenure 5 --balance 76000 --num-of-products 1 \
        --has-cr-card 1 --is-active-member 1 --estimated-salary 100000

It uses the same model.keras and preprocessor.pkl as app.py, builds the same
one-row table (same column names and order), and prints the churn probability
and a STAY/LEAVE verdict (LEAVE when probability >= threshold, default 0.50).

Bad input is reported as a one-line error and a non-zero exit code:
  2 = bad command-line arguments, 1 = missing/unloadable model or preprocessor.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Same ranges as the inputs in app.py
INT_FIELDS = {  # dest: (column name, min, max)
    "credit_score": ("CreditScore", 300, 900),
    "age": ("Age", 18, 92),
    "tenure": ("Tenure", 0, 10),
    "num_of_products": ("NumOfProducts", 1, 4),
}
FLOAT_FIELDS = {
    "balance": ("Balance", 0.0, 260000.0),
    "estimated_salary": ("EstimatedSalary", 0.0, 200000.0),
}
DEFAULT_THRESHOLD = 0.50  # same default as app.py
THRESHOLD_RANGE = (0.30, 0.70)  # same slider range as app.py


class CliError(Exception):
    """An error with a message meant for the user (no traceback)."""

    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def ranged(kind, lo, hi):
    """argparse type: parse a number of `kind` and check lo <= value <= hi."""
    def parse(text):
        try:
            value = kind(text)
        except ValueError:
            word = "a whole number" if kind is int else "a number"
            raise argparse.ArgumentTypeError(f"{text!r} is not {word}")
        if value != value or not lo <= value <= hi:  # value != value catches NaN
            raise argparse.ArgumentTypeError(f"{text} is out of range ({lo} to {hi})")
        return value
    return parse


def yes_no(text):
    """argparse type for HasCrCard / IsActiveMember: 1/0 or yes/no -> 1/0."""
    t = text.strip().lower()
    if t in ("1", "yes", "y", "true"):
        return 1
    if t in ("0", "no", "n", "false"):
        return 0
    raise argparse.ArgumentTypeError(f"{text!r} must be 1 or 0 (or yes/no)")


def build_parser():
    p = argparse.ArgumentParser(
        description="Predict how likely one bank customer is to leave (churn).")
    req = p.add_argument_group("customer details (all required)")
    req.add_argument("--credit-score", required=True, type=ranged(int, 300, 900),
                     help="CreditScore, whole number 300-900")
    req.add_argument("--geography", required=True,
                     help="Geography, one the model knows (France, Germany, Spain)")
    req.add_argument("--gender", required=True,
                     help="Gender, one the model knows (Male, Female)")
    req.add_argument("--age", required=True, type=ranged(int, 18, 92),
                     help="Age in years, 18-92")
    req.add_argument("--tenure", required=True, type=ranged(int, 0, 10),
                     help="Tenure: years with the bank, 0-10")
    req.add_argument("--balance", required=True, type=ranged(float, 0.0, 260000.0),
                     help="Account balance, 0-260000")
    req.add_argument("--num-of-products", required=True, type=ranged(int, 1, 4),
                     help="NumOfProducts: number of bank products, 1-4")
    req.add_argument("--has-cr-card", required=True, type=yes_no,
                     help="HasCrCard: 1 = has a credit card, 0 = not (yes/no also accepted)")
    req.add_argument("--is-active-member", required=True, type=yes_no,
                     help="IsActiveMember: 1 = active, 0 = not (yes/no also accepted)")
    req.add_argument("--estimated-salary", required=True,
                     type=ranged(float, 0.0, 200000.0),
                     help="EstimatedSalary, 0-200000")
    opt = p.add_argument_group("options")
    opt.add_argument("--threshold", type=ranged(float, *THRESHOLD_RANGE),
                     default=DEFAULT_THRESHOLD,
                     help="LEAVE when probability >= this (default 0.50, range 0.30-0.70, as in the app)")
    opt.add_argument("--model", type=Path, default=HERE / "model.keras",
                     help="path to the trained model (default: model.keras next to this script)")
    opt.add_argument("--preprocessor", type=Path, default=HERE / "preprocessor.pkl",
                     help="path to the fitted preprocessor (default: preprocessor.pkl next to this script)")
    return p


def known_categories(preprocessor):
    """{column: [categories]} read from the fitted OneHotEncoder."""
    known = {}
    for _name, transformer, columns in preprocessor.transformers_:
        cats = getattr(transformer, "categories_", None)
        if cats is not None:
            for col, values in zip(columns, cats):
                known[col] = [str(v) for v in values]
    return known


def match_category(column, value, known):
    """Return the preprocessor's spelling of `value` (case-insensitive), or fail."""
    options = known.get(column)
    if options is None:
        return value
    for option in options:
        if option.lower() == value.strip().lower():
            return option
    raise CliError(f"unknown {column} {value!r}; expected one of: {', '.join(options)}", code=2)


def load_preprocessor(path):
    if not path.is_file():
        raise CliError(f"preprocessor file not found: {path}")
    import joblib
    try:
        return joblib.load(path)
    except Exception as exc:  # corrupt or incompatible file
        raise CliError(f"could not load preprocessor {path}: {exc}")


def load_model(path):
    if not path.is_file():
        raise CliError(f"model file not found: {path}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # keep TensorFlow quiet
    try:
        from tensorflow import keras
        return keras.models.load_model(path)
    except Exception as exc:
        raise CliError(f"could not load model {path}: {exc}")


def predict(args):
    # Check both files exist before the slow TensorFlow import
    for path, what in ((args.preprocessor, "preprocessor"), (args.model, "model")):
        if not path.is_file():
            raise CliError(f"{what} file not found: {path}")

    preprocessor = load_preprocessor(args.preprocessor)
    known = known_categories(preprocessor)
    geography = match_category("Geography", args.geography, known)
    gender = match_category("Gender", args.gender, known)
    model = load_model(args.model)

    import pandas as pd
    # Same one-row table as app.py: same column names, same order
    customer = pd.DataFrame([{
        "CreditScore": args.credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": args.age,
        "Tenure": args.tenure,
        "Balance": args.balance,
        "NumOfProducts": args.num_of_products,
        "HasCrCard": args.has_cr_card,
        "IsActiveMember": args.is_active_member,
        "EstimatedSalary": args.estimated_salary,
    }])
    X_new = preprocessor.transform(customer)
    return float(model.predict(X_new, verbose=0)[0][0])


def main(argv=None):
    args = build_parser().parse_args(argv)  # exits 2 with a message on bad arguments
    try:
        probability = predict(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    verdict = "LEAVE" if probability >= args.threshold else "STAY"
    print(f"Churn probability: {probability * 100:.1f}% ({probability:.4f})")
    print(f"Verdict: likely to {verdict} (threshold {args.threshold:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
