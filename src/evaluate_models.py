"""
evaluate_models.py
-------------------
Extracts feature importance from the tree-based models (RandomForest) for
both course-level targets and saves comparison/importance charts.

Feature importance here reflects the RandomForest's internal impurity-based
importance - it measures association/predictive usefulness within the
model, NOT causation. This is stated explicitly wherever importances are
reported (Phase 11 requirement).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from train_models import build_preprocessor, NUMERIC_FEATURES_BASE, CATEGORICAL_FEATURES, RANDOM_STATE

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def get_feature_names(preprocessor):
    num_names = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_names = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    return list(num_names) + cat_names


def compute_and_plot_importance(course_df: pd.DataFrame, target: str, out_name: str, title: str):
    X = course_df[NUMERIC_FEATURES_BASE + CATEGORICAL_FEATURES].copy()
    y = course_df[target].copy()

    preprocessor = build_preprocessor(NUMERIC_FEATURES_BASE, CATEGORICAL_FEATURES)
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", rf)])
    pipe.fit(X, y)

    feature_names = get_feature_names(pipe.named_steps["preprocessor"])
    importances = pipe.named_steps["model"].feature_importances_

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"][:12][::-1], imp_df["importance"][:12][::-1], color="#4C72B0")
    plt.title(title)
    plt.xlabel("RandomForest impurity-based importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, out_name), dpi=110)
    plt.close()

    return imp_df


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_all
    from feature_engineering import build_course_level_dataset, add_engineered_features

    raw = load_raw_data()
    processed = preprocess_all(raw)
    course_df = build_course_level_dataset(processed["courses"], processed["transactions"], processed["teachers"])
    course_df = add_engineered_features(course_df)

    print("=== Feature importance for EnrollmentCount (demand) ===")
    enroll_imp = compute_and_plot_importance(
        course_df, "EnrollmentCount", "enrollment_feature_importance.png",
        "Feature Importance - Enrollment (Demand) Prediction"
    )
    print(enroll_imp.to_string(index=False))

    print("\n=== Feature importance for TotalRevenue ===")
    revenue_imp = compute_and_plot_importance(
        course_df, "TotalRevenue", "revenue_feature_importance.png",
        "Feature Importance - Revenue Prediction"
    )
    print(revenue_imp.to_string(index=False))

    enroll_imp.to_csv(os.path.join(REPORTS_DIR, "enrollment_feature_importance.csv"), index=False)
    revenue_imp.to_csv(os.path.join(REPORTS_DIR, "revenue_feature_importance.csv"), index=False)
    print("\nSaved feature importance charts and CSVs to reports/")
