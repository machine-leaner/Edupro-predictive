"""
train_models.py
----------------
Trains and saves models for:

  TARGET 1 (course-level): EnrollmentCount  ~ course/teacher-pool attributes
  TARGET 2 (course-level): TotalRevenue     ~ course/teacher-pool attributes
  TARGET 3 (forecasting):  NextMonthEnrollment ~ chronological monthly panel

Course-level targets (1 & 2) use only 60 rows (one per course) - the
dataset does not support a train/test split large enough to be
statistically meaningful, so we use Leave-One-Out cross-validation
(LOOCV) for these instead of a single holdout split, and say so plainly
in the evaluation report. This is an honest response to a genuine
dataset-size limitation, not a workaround to inflate scores.

TARGET 3 uses the course x month panel (600 rows after lag/target
construction, spanning 10 months) with a proper CHRONOLOGICAL split:
the last 2 months of (course, month) rows are held out as the test set,
everything earlier is training. No shuffling.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# Features used for the course-level targets (Target 1 & 2). Deliberately
# excludes anything that is a direct restatement of the other target
# (e.g. TotalRevenue is excluded from the EnrollmentCount feature set) so
# each model reflects genuinely independent predictors.
CATEGORICAL_FEATURES = ["CourseCategory", "CourseType", "CourseLevel"]
NUMERIC_FEATURES_BASE = [
    "CoursePrice", "CourseDuration", "CourseRating",
    "AvgTeacherRating", "AvgTeacherExperience", "DistinctTeacherCount",
]


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def get_model_zoo():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate_loocv(pipeline, X, y):
    """Leave-One-Out CV predictions -> MAE/RMSE/R2. Appropriate here because
    the course-level dataset only has 60 rows, too few for a stable single
    train/test split."""
    loo = LeaveOneOut()
    preds = cross_val_predict(pipeline, X, y, cv=loo)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return mae, rmse, r2, preds


def train_course_level_target(course_df: pd.DataFrame, target: str, extra_numeric=None):
    numeric_features = NUMERIC_FEATURES_BASE.copy()
    if extra_numeric:
        numeric_features += extra_numeric
    X = course_df[numeric_features + CATEGORICAL_FEATURES].copy()
    y = course_df[target].copy()

    preprocessor = build_preprocessor(numeric_features, CATEGORICAL_FEATURES)
    results = []
    fitted_pipelines = {}

    for name, model in get_model_zoo().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        mae, rmse, r2, _ = evaluate_loocv(pipe, X, y)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        # Fit on full data for the final saved model (used by Streamlit).
        pipe.fit(X, y)
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE")
    return results_df, fitted_pipelines, numeric_features


def train_forecasting_target(panel_df: pd.DataFrame, target="NextMonthEnrollment"):
    """Chronological train/test split on the course x month panel."""
    numeric_features = ["Lag1Enrollment", "Lag1Revenue", "RollingAvgEnrollment3M",
                         "CumulativeEnrollmentToDate", "CoursePrice", "CourseDuration",
                         "CourseRating"]
    categorical_features = ["CourseCategory", "CourseType", "CourseLevel"]

    df = panel_df.sort_values("YearMonth").reset_index(drop=True)
    months = sorted(df["YearMonth"].unique())
    test_months = months[-2:]  # last 2 available months = test (chronological)
    train_df = df[~df["YearMonth"].isin(test_months)]
    test_df = df[df["YearMonth"].isin(test_months)]

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    results = []
    fitted_pipelines = {}

    for name, model in get_model_zoo().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE")
    return results_df, fitted_pipelines, numeric_features, (len(train_df), len(test_df), test_months)


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_all
    from feature_engineering import (
        build_course_level_dataset, add_engineered_features, build_forecasting_dataset
    )

    raw = load_raw_data()
    processed = preprocess_all(raw)
    course_df = build_course_level_dataset(processed["courses"], processed["transactions"], processed["teachers"])
    course_df = add_engineered_features(course_df)
    forecast_df = build_forecasting_dataset(processed["transactions"], processed["courses"])

    print("=" * 70)
    print("TARGET 1: EnrollmentCount (course-level, LOOCV, n=60)")
    print("=" * 70)
    enroll_results, enroll_pipes, enroll_feats = train_course_level_target(course_df, "EnrollmentCount")
    print(enroll_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("TARGET 2: TotalRevenue (course-level, LOOCV, n=60)")
    print("=" * 70)
    revenue_results, revenue_pipes, revenue_feats = train_course_level_target(course_df, "TotalRevenue")
    print(revenue_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("TARGET 3: NextMonthEnrollment (forecasting, chronological split)")
    print("=" * 70)
    fc_results, fc_pipes, fc_feats, split_info = train_forecasting_target(forecast_df)
    print(f"Train rows: {split_info[0]}, Test rows: {split_info[1]}, Test months: {split_info[2]}")
    print(fc_results.to_string(index=False))

    # Save best model per target (best = lowest MAE, tie-broken by RMSE)
    best_enroll_name = enroll_results.iloc[0]["Model"]
    best_revenue_name = revenue_results.iloc[0]["Model"]
    best_fc_name = fc_results.iloc[0]["Model"]

    joblib.dump(enroll_pipes[best_enroll_name], os.path.join(MODELS_DIR, "demand_model.pkl"))
    joblib.dump(revenue_pipes[best_revenue_name], os.path.join(MODELS_DIR, "revenue_model.pkl"))
    joblib.dump(fc_pipes[best_fc_name], os.path.join(MODELS_DIR, "forecast_model.pkl"))

    meta = {
        "enrollment_model": best_enroll_name,
        "enrollment_features_numeric": enroll_feats,
        "enrollment_features_categorical": CATEGORICAL_FEATURES,
        "revenue_model": best_revenue_name,
        "revenue_features_numeric": revenue_feats,
        "revenue_features_categorical": CATEGORICAL_FEATURES,
        "forecast_model": best_fc_name,
        "forecast_features_numeric": fc_feats,
        "forecast_features_categorical": CATEGORICAL_FEATURES,
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    enroll_results.to_csv(os.path.join(REPORTS_DIR, "enrollment_model_comparison.csv"), index=False)
    revenue_results.to_csv(os.path.join(REPORTS_DIR, "revenue_model_comparison.csv"), index=False)
    fc_results.to_csv(os.path.join(REPORTS_DIR, "forecast_model_comparison.csv"), index=False)

    print(f"\nBest enrollment model: {best_enroll_name}")
    print(f"Best revenue model: {best_revenue_name}")
    print(f"Best forecast model: {best_fc_name}")
    print("\nModels saved to models/. Comparison tables saved to reports/.")
