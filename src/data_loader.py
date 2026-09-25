"""
data_loader.py
---------------
Loads the raw EduPro Online Platform workbook. Never writes back to the
source file - the raw workbook is treated as read-only ground truth.
"""

import os
import pandas as pd

# Default path to the raw workbook (configurable via argument or env var)
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "EduPro_Online_Platform.xlsx",
)


def load_raw_data(path: str = DEFAULT_DATA_PATH) -> dict:
    """
    Load all four sheets of the EduPro workbook into a dict of DataFrames.

    Returns
    -------
    dict with keys: 'users', 'teachers', 'courses', 'transactions'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    xl = pd.ExcelFile(path)
    required_sheets = {"Users", "Teachers", "Courses", "Transactions"}
    missing = required_sheets - set(xl.sheet_names)
    if missing:
        raise ValueError(f"Workbook is missing expected sheet(s): {missing}")

    data = {
        "users": pd.read_excel(xl, sheet_name="Users"),
        "teachers": pd.read_excel(xl, sheet_name="Teachers"),
        "courses": pd.read_excel(xl, sheet_name="Courses"),
        "transactions": pd.read_excel(xl, sheet_name="Transactions"),
    }
    return data


def data_quality_summary(data: dict) -> pd.DataFrame:
    """
    Produce a compact data-quality summary table across all sheets:
    rows, columns, missing values, duplicate rows, unique key count.
    """
    key_cols = {
        "users": "UserID",
        "teachers": "TeacherID",
        "courses": "CourseID",
        "transactions": "TransactionID",
    }
    rows = []
    for name, df in data.items():
        key = key_cols[name]
        rows.append(
            {
                "sheet": name,
                "n_rows": len(df),
                "n_columns": df.shape[1],
                "missing_values_total": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "unique_keys": df[key].nunique(),
                "key_is_unique": df[key].nunique() == len(df),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = load_raw_data()
    for name, df in raw.items():
        print(f"{name}: shape={df.shape}")
    print()
    print(data_quality_summary(raw).to_string(index=False))
