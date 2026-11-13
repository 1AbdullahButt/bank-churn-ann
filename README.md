# Customer Churn Predictor (ANN + Streamlit)

**Live app:** https://bank-churn-ann.streamlit.app/

An Artificial Neural Network (Keras) that predicts whether a bank customer will **leave the bank**, and a small Streamlit web app that uses it.

**Dataset:** `Churn_Modelling.csv` (10,000 customers, target `Exited`: 1 = left, 0 = stayed).

## Results (test set, 2,000 customers)
| Metric | Value |
|---|---|
| Accuracy | 85.95% (an "always stays" model gets 79.6%) |
| Class-1 (leavers) recall | 0.47 |
| Class-1 precision | 0.74 |

The model misses a bit more than half of the customers who leave. In the bonus part of the notebook, `class_weight` (class 1 counts 3x) raises recall to 0.73 but lowers accuracy to 81.2%.

## Files
| File | What it is |
|---|---|
| `train.ipynb` | Steps 0 to 9: load, clean, split, preprocess, build, train, evaluate, save, app (with "My understanding" notes) |
| `app.py` | Streamlit app |
| `model.keras` | trained ANN (13 inputs, 16 and 8 ReLU neurons, 1 Sigmoid output) |
| `preprocessor.pkl` | fitted One-Hot Encoder + StandardScaler (needed by the app) |
| `requirements.txt` | libraries |
| `loss_plot.png`, `confusion_matrix.png` | plots from the notebook |
| `screenshots/` | the app showing one STAY and one LEAVE prediction |

## Run the app on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
(If `streamlit` is not found, use `python -m streamlit run app.py`.)

The two files were created with Python 3.13, TensorFlow 2.21, Keras 3.15 and scikit-learn 1.9.1. If loading fails on an older setup, re-run `train.ipynb` to make fresh `model.keras` and `preprocessor.pkl` files.

## Retrain
Open `train.ipynb` in Jupyter or Google Colab and run it from top to bottom (in Colab, upload `Churn_Modelling.csv` when asked). The last Step 9 cell can start the app from Colab.

## Deployment (bonus)
The app is deployed free on Streamlit Community Cloud: https://bank-churn-ann.streamlit.app/

To deploy your own copy:
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create a new app, pick this repository, branch `main`, main file `app.py`.
3. Deploy. Streamlit installs `requirements.txt` for you.
