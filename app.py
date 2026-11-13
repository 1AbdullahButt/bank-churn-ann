"""Bank customer churn predictor (Streamlit app).

Run it with:   streamlit run app.py
It needs model.keras and preprocessor.pkl in the SAME folder as this file.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

HERE = Path(__file__).parent  # the folder where app.py lives

st.set_page_config(page_title="Churn Predictor", page_icon="🏦")


@st.cache_resource
def load_files():
    """Load the trained model and the preprocessor ONE time, when the app starts."""
    model = keras.models.load_model(HERE / "model.keras")
    preprocessor = joblib.load(HERE / "preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_files()

st.title("🏦 Bank Customer Churn Predictor")
st.write("Fill in the customer's details, then click **Predict** to see how likely "
         "this customer is to leave the bank.")

# ---------- 1. Inputs (one for every feature the model was trained on) ----------
left, right = st.columns(2)
with left:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 92, 38)
    credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650, step=10)
    tenure = st.slider("Tenure (years with the bank)", 0, 10, 5)
with right:
    balance = st.number_input("Balance", min_value=0.0, max_value=260000.0, value=76000.0, step=1000.0)
    num_products = st.selectbox("Number of products", [1, 2, 3, 4])
    has_card = st.radio("Has a credit card?", ["Yes", "No"], horizontal=True)
    is_active = st.radio("Is an active member?", ["Yes", "No"], horizontal=True)
    salary = st.number_input("Estimated salary", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)

# Bonus: the user can move the decision threshold (default 0.5, as in the manual)
threshold = st.slider("Decision threshold", 0.30, 0.70, 0.50, step=0.01,
                      help="The customer is called 'likely to LEAVE' when the churn probability "
                           "is at or above this value. Lower = catches more leavers, but more false alarms.")

# ---------- 2. Predict ----------
inputs = (geography, gender, age, credit_score, tenure, balance,
          num_products, has_card, is_active, salary)

if st.button("Predict", type="primary"):
    # Build a ONE-row table with the SAME column names the model was trained on
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,          # Yes/No back to 1/0
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }])
    X_new = preprocessor.transform(customer)                   # same encoding + scaling as training
    probability = float(model.predict(X_new, verbose=0)[0][0])
    st.session_state["result"] = {"probability": probability, "inputs": inputs}

# ---------- 3. Show the result ----------
result = st.session_state.get("result")
if result is not None:
    if result["inputs"] != inputs:
        st.warning("You changed the details. Click **Predict** again to update the result.")
    else:
        probability = result["probability"]
        st.metric("Churn probability", f"{probability * 100:.1f}%")
        if probability >= threshold:
            st.error("⚠️ This customer is likely to LEAVE the bank.")
        else:
            st.success("✅ This customer is likely to STAY with the bank.")
        st.caption(f"Verdict uses a threshold of {threshold:.2f}.")
