"""Smoke tests: the saved preprocessor and model load and work on one customer row."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

COLUMNS = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
           "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]


@pytest.fixture(scope="module")
def customer():
    row = pd.DataFrame([{
        "CreditScore": 650, "Geography": "France", "Gender": "Male", "Age": 38,
        "Tenure": 5, "Balance": 76000.0, "NumOfProducts": 2, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 100000.0,
    }])
    assert list(row.columns) == COLUMNS
    return row


@pytest.fixture(scope="module")
def preprocessor():
    return joblib.load(ROOT / "preprocessor.pkl")


@pytest.fixture(scope="module")
def model():
    from tensorflow import keras
    return keras.models.load_model(ROOT / "model.keras")


def test_preprocessor_transforms_one_row(preprocessor, customer):
    X = np.asarray(preprocessor.transform(customer))
    assert X.shape[0] == 1
    assert X.shape[1] == 13  # 8 numeric + 3 Geography + 2 Gender (one-hot)
    assert np.isfinite(X).all()


def test_model_predicts_probability(model, preprocessor, customer):
    X = preprocessor.transform(customer)
    prediction = model.predict(X, verbose=0)
    assert prediction.shape == (1, 1)
    probability = float(prediction[0][0])
    assert 0.0 <= probability <= 1.0
