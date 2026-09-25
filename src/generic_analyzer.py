"""
generic_analyzer.py
--------------------
Reusable, dataset-agnostic analysis functions that power the "Upload &
Analyze Dataset" page in app.py. Nothing here is EduPro-specific - it
works on any uploaded CSV/Excel file.

Kept separate from the EduPro-specific pipeline (data_loader.py,
preprocessing.py, feature_engineering.py, train_models.py) so the two
modes never interfere with each other and the existing EduPro dashboard
code is untouched.
"""

import io
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

RANDOM_STATE = 42

TARGET_NAME_HINTS = [
    "target", "sales", "revenue", "amount", "enrollment", "demand",
    "score", "quantity", "count", "price", "profit", "churn", "label",
    "outcome", "class",
]


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------

class UnsupportedFileError(Exception):
    pass


class EmptyDatasetError(Exception):
    pass


def get_excel_sheet_names(file_bytes: bytes) -> list:
    """Return sheet names of an uploaded Excel file, or raise a clear
    error if the workbook can't be read."""
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise UnsupportedFileError(
            "This Excel file could not be opened - it may be corrupted or "
            "in an unsupported format."
        ) from e
    if not xl.sheet_names:
        raise EmptyDatasetError("This workbook has no sheets to read.")
    return xl.sheet_names


def load_uploaded_file(uploaded_file, sheet_name=None) -> pd.DataFrame:
    """
    Load an uploaded CSV/XLS/XLSX file into a DataFrame. Does not modify
    or persist the original uploaded bytes.

    Raises UnsupportedFileError / EmptyDatasetError with user-friendly
    messages on failure rather than letting a raw traceback bubble up.
    """
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) == 0:
        raise EmptyDatasetError("The uploaded file is empty (0 bytes).")

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except pd.errors.EmptyDataError:
            raise EmptyDatasetError("This CSV file has no columns/data to parse.")
        except Exception as e:
            raise UnsupportedFileError(
                "This CSV file could not be parsed - check that it is a "
                "valid, comma-delimited text file."
            ) from e
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        except Exception as e:
            raise UnsupportedFileError(
                "This Excel file/sheet could not be read - it may be "
                "corrupted or the selected sheet may be empty."
            ) from e
    else:
        raise UnsupportedFileError(
            f"Unsupported file type for '{uploaded_file.name}'. "
            "Please upload a .csv, .xlsx, or .xls file."
        )

    if df is None or df.shape[1] == 0:
        raise EmptyDatasetError("No usable columns were found in this file.")
    if df.shape[0] == 0:
        raise EmptyDatasetError("This file has columns but no data rows.")

    return df


# ----------------------------------------------------------------------
# 2. Basic info / data quality
# ----------------------------------------------------------------------

def get_basic_info(df: pd.DataFrame) -> dict:
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    }


def data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        missing = df[col].isna().sum()
        rows.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Missing": int(missing),
            "Missing %": round(100 * missing / n, 2) if n else 0.0,
            "Unique": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def data_quality_summary(df: pd.DataFrame) -> dict:
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
    }


# ----------------------------------------------------------------------
# 3. Column type detection
# ----------------------------------------------------------------------

ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|id$|Id$|ID$)", re.IGNORECASE)


def _looks_like_id_name(col: str) -> bool:
    lower = col.lower()
    return lower == "id" or lower.endswith("id") or lower.endswith("_id") or lower.startswith("id_")


def _try_parse_dates(series: pd.Series, sample_size: int = 200) -> bool:
    """Heuristic: does this object column look like dates? Sample a
    subset (for speed on large datasets) and require a high parse rate."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.sample(min(sample_size, len(non_null)), random_state=RANDOM_STATE)
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
        except Exception:
            return False
    success_rate = parsed.notna().mean()
    return success_rate >= 0.9


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Classify every column into exactly one bucket: id, constant, date,
    numeric, or categorical (priority in that order). Returns a dict of
    lists plus a per-column 'primary_type' mapping for display.
    """
    n = len(df)
    id_cols, constant_cols, date_cols, numeric_cols, categorical_cols = [], [], [], [], []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)

        # Constant column (all one value, or all missing)
        if nunique <= 1:
            constant_cols.append(col)
            continue

        # ID-like column: name looks like an ID AND is (near-)unique, or
        # is fully unique regardless of name (common for primary keys).
        is_unique_ish = nunique >= 0.95 * n and n > 1
        if (_looks_like_id_name(col) and is_unique_ish) or (is_unique_ish and series.dtype == object):
            id_cols.append(col)
            continue

        # Date column: already datetime, or object column that parses well.
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue
        if series.dtype == object and _try_parse_dates(series):
            date_cols.append(col)
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        # Everything else: categorical
        categorical_cols.append(col)

    return {
        "id_cols": id_cols,
        "constant_cols": constant_cols,
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


# ----------------------------------------------------------------------
# 4. Correlation
# ----------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr()


def strong_correlations(corr: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({"Feature 1": cols[i], "Feature 2": cols[j], "Correlation": round(val, 3)})
    return pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False) if pairs else pd.DataFrame(
        columns=["Feature 1", "Feature 2", "Correlation"]
    )


# ----------------------------------------------------------------------
# 5. Outlier detection
# ----------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return mask.sum(), lower, upper


def detect_outliers_zscore(df: pd.DataFrame, col: str, z_thresh: float = 3.0):
    s = df[col].dropna()
    if s.std(ddof=0) == 0 or len(s) < 2:
        return 0
    z = (s - s.mean()) / s.std(ddof=0)
    return int((z.abs() > z_thresh).sum())


def outlier_summary(df: pd.DataFrame, numeric_cols: list, method: str = "IQR") -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in numeric_cols:
        if method == "IQR":
            count, lower, upper = detect_outliers_iqr(df, col)
        else:
            count = detect_outliers_zscore(df, col)
        rows.append({
            "Column": col,
            "Potential Outliers": int(count),
            "Percentage": round(100 * count / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("Potential Outliers", ascending=False)


def cap_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Winsorize a column to its IQR fences."""
    out = df.copy()
    _, lower, upper = detect_outliers_iqr(df, col)
    out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    _, lower, upper = detect_outliers_iqr(df, col)
    return df[(df[col] >= lower) & (df[col] <= upper)]


# ----------------------------------------------------------------------
# 6. Missing value handling
# ----------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing == 0:
            continue
        if col in numeric_cols:
            recommended = "Median imputation (robust to outliers)"
        elif col in categorical_cols:
            recommended = "Mode, or 'Unknown' category"
        else:
            recommended = "Review manually"
        rows.append({
            "Column": col,
            "Missing Count": int(missing),
            "Missing %": round(100 * missing / n, 2) if n else 0.0,
            "Recommended Handling": recommended,
        })
    return pd.DataFrame(rows)


def apply_missing_value_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    """strategy in {'drop_rows', 'mean', 'median', 'mode', 'ffill', 'unknown'}"""
    out = df.copy()
    if strategy == "drop_rows":
        out = out[out[col].notna()]
    elif strategy == "mean":
        out[col] = out[col].fillna(out[col].mean())
    elif strategy == "median":
        out[col] = out[col].fillna(out[col].median())
    elif strategy == "mode":
        mode = out[col].mode()
        if len(mode) > 0:
            out[col] = out[col].fillna(mode.iloc[0])
    elif strategy == "ffill":
        out[col] = out[col].ffill()
    elif strategy == "unknown":
        out[col] = out[col].astype(object).fillna("Unknown")
    return out


# ----------------------------------------------------------------------
# 7. Target suggestion
# ----------------------------------------------------------------------

def suggest_target_columns(df: pd.DataFrame, numeric_cols: list) -> list:
    suggestions = []
    for col in numeric_cols:
        lower = col.lower()
        if any(hint in lower for hint in TARGET_NAME_HINTS):
            suggestions.append(col)
    # Fall back: if no name-based hint matched anything, offer all numeric
    # columns as candidates (still never auto-selected).
    return suggestions if suggestions else numeric_cols


def suggest_problem_type(df: pd.DataFrame, target_col: str) -> str:
    series = df[target_col].dropna()
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique()
        # Few distinct values relative to row count -> looks categorical/class-like.
        if nunique <= min(10, max(2, int(0.05 * len(series)))):
            return "Classification"
        return "Regression"
    return "Classification"


# ----------------------------------------------------------------------
# 8/9. Preprocessing + modeling
# ----------------------------------------------------------------------

def build_feature_lists(df: pd.DataFrame, target_col: str, col_types: dict, extract_date_parts=True):
    """Determine final numeric/categorical feature lists for modeling,
    excluding the target, ID columns, and constant columns. Optionally
    expands date columns into Year/Month/Day/DayOfWeek numeric features."""
    numeric_features = [c for c in col_types["numeric_cols"] if c != target_col]
    categorical_features = [c for c in col_types["categorical_cols"] if c != target_col]
    date_features = [c for c in col_types["date_cols"] if c != target_col]

    working_df = df.copy()
    derived_numeric = []
    if extract_date_parts:
        for c in date_features:
            parsed = pd.to_datetime(working_df[c], errors="coerce")
            working_df[f"{c}_Year"] = parsed.dt.year
            working_df[f"{c}_Month"] = parsed.dt.month
            working_df[f"{c}_Day"] = parsed.dt.day
            working_df[f"{c}_DayOfWeek"] = parsed.dt.dayofweek
            derived_numeric += [f"{c}_Year", f"{c}_Month", f"{c}_Day", f"{c}_DayOfWeek"]
    numeric_features = numeric_features + derived_numeric

    return working_df, numeric_features, categorical_features


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers=transformers)


def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_and_evaluate(
    df: pd.DataFrame, target_col: str, numeric_features: list, categorical_features: list,
    problem_type: str, test_size: float = 0.2,
):
    """
    Trains every model appropriate for problem_type, evaluates on a held-out
    test split, and returns (results_df, fitted_pipelines, X_test, y_test,
    split_note).
    """
    feature_cols = numeric_features + categorical_features
    data = df[feature_cols + [target_col]].dropna(subset=[target_col])
    X = data[feature_cols]
    y = data[target_col]

    if len(data) < 10:
        raise ValueError(
            "Too few rows with a non-missing target (fewer than 10) to "
            "train/evaluate a model reliably."
        )

    stratify = y if (problem_type == "Classification" and y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = get_regression_models() if problem_type == "Regression" else get_classification_models()

    results = []
    fitted_pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if problem_type == "Regression":
            results.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "R2": r2_score(y_test, preds),
            })
        else:
            if y.nunique() == 2:
                avg = "binary"
                pos_label = sorted(y.unique())[-1]
                kw = {"average": avg, "pos_label": pos_label, "zero_division": 0}
            else:
                kw = {"average": "macro", "zero_division": 0}
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, **kw),
                "Recall": recall_score(y_test, preds, **kw),
                "F1 Score": f1_score(y_test, preds, **kw),
            })
        fitted_pipelines[name] = pipe

    sort_col = "MAE" if problem_type == "Regression" else "F1 Score"
    ascending = problem_type == "Regression"
    results_df = pd.DataFrame(results).sort_values(sort_col, ascending=ascending)

    return results_df, fitted_pipelines, X_test, y_test, (len(X_train), len(X_test))


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list:
    names = []
    for trans_name, trans, cols in preprocessor.transformers_:
        if trans_name == "num":
            names += list(cols)
        elif trans_name == "cat":
            encoder = trans.named_steps["encoder"]
            names += list(encoder.get_feature_names_out(cols))
    return names


def compute_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Works for any tree-based model in the pipeline (RandomForest /
    GradientBoosting). Returns empty DataFrame if the model has no
    feature_importances_ attribute (e.g. Linear/Logistic Regression)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    feature_names = get_feature_names_from_preprocessor(pipeline.named_steps["preprocessor"])
    importances = model.feature_importances_
    return pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )


def confusion_matrix_df(y_test, preds) -> pd.DataFrame:
    labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(preds)])))
    cm = confusion_matrix(y_test, preds, labels=labels)
    return pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
