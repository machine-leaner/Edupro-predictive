"""
predictions.py
---------------
Thin wrapper around the saved model pipelines so the Streamlit app (and
anything else) can load once and predict without retraining.
"""

import os
import json
import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_models():
    demand_model = joblib.load(os.path.join(MODELS_DIR, "demand_model.pkl"))
    revenue_model = joblib.load(os.path.join(MODELS_DIR, "revenue_model.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return demand_model, revenue_model, meta


def predict_enrollment(demand_model, meta, inputs: dict) -> float:
    cols = meta["enrollment_features_numeric"] + meta["enrollment_features_categorical"]
    row = pd.DataFrame([{c: inputs[c] for c in cols}])
    return float(demand_model.predict(row)[0])


def predict_revenue(revenue_model, meta, inputs: dict) -> float:
    cols = meta["revenue_features_numeric"] + meta["revenue_features_categorical"]
    row = pd.DataFrame([{c: inputs[c] for c in cols}])
    return float(revenue_model.predict(row)[0])
