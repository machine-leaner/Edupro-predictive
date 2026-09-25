"""
preprocessing.py
-----------------
Cleans the raw EduPro sheets and produces validated, typed DataFrames.

Design notes (documented per project requirements):

* No missing values or duplicate rows were found in any sheet during Phase 1
  inspection, so no imputation or de-duplication logic is exercised on this
  dataset - but the checks are still implemented defensively so the pipeline
  is safe to re-run if the source data changes.
* TransactionDate is cast to datetime64 (it parses cleanly, 0 failures).
* CourseType/CoursePrice==0 cleanly identifies free enrollments; Amount==0
  transactions are exactly the free-course enrollments (verified in Phase 1).
  We keep free enrollments IN the dataset for demand modeling (they are real
  enrollments) but exclude them from "paid revenue" style aggregations,
  since a free enrollment contributes 0 revenue by construction.
* The raw dict returned by data_loader is never mutated in place; every
  function here returns new DataFrames so the raw snapshot stays intact
  for auditing.
"""

import os
import pandas as pd
import numpy as np


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    df = df.drop_duplicates(subset="UserID")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Gender"] = df["Gender"].astype("category")
    return df


def clean_teachers(teachers: pd.DataFrame) -> pd.DataFrame:
    df = teachers.copy()
    df = df.drop_duplicates(subset="TeacherID")
    df["YearsOfExperience"] = pd.to_numeric(df["YearsOfExperience"], errors="coerce")
    df["TeacherRating"] = pd.to_numeric(df["TeacherRating"], errors="coerce")
    df["Expertise"] = df["Expertise"].astype("category")
    df["Gender"] = df["Gender"].astype("category")
    # Sanity clip: ratings are documented 1-5 scale; guard against any future
    # out-of-range values without silently fabricating data.
    out_of_range = ~df["TeacherRating"].between(0, 5)
    if out_of_range.any():
        raise ValueError(
            f"{out_of_range.sum()} TeacherRating value(s) fall outside [0,5] - "
            "investigate before proceeding."
        )
    return df


def clean_courses(courses: pd.DataFrame) -> pd.DataFrame:
    df = courses.copy()
    df = df.drop_duplicates(subset="CourseID")
    df["CoursePrice"] = pd.to_numeric(df["CoursePrice"], errors="coerce")
    df["CourseDuration"] = pd.to_numeric(df["CourseDuration"], errors="coerce")
    df["CourseRating"] = pd.to_numeric(df["CourseRating"], errors="coerce")
    for col in ["CourseCategory", "CourseType", "CourseLevel"]:
        df[col] = df[col].astype("category")

    if (df["CoursePrice"] < 0).any():
        raise ValueError("Negative CoursePrice found - investigate before proceeding.")

    # Cross-check CourseType against price, since both encode "free vs paid".
    mismatch = df[(df["CourseType"] == "Free") & (df["CoursePrice"] > 0)]
    if len(mismatch) > 0:
        raise ValueError(
            f"{len(mismatch)} course(s) marked 'Free' but have CoursePrice > 0 - "
            "investigate before proceeding."
        )

    df["IsFreeCourse"] = df["CoursePrice"] == 0
    return df


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df = df.drop_duplicates(subset="TransactionID")

    # TransactionDate parses cleanly (verified in Phase 1); coerce defensively.
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    n_bad_dates = df["TransactionDate"].isna().sum()
    if n_bad_dates > 0:
        raise ValueError(
            f"{n_bad_dates} TransactionDate value(s) failed to parse - "
            "investigate before proceeding."
        )

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if (df["Amount"] < 0).any():
        raise ValueError("Negative transaction Amount found - investigate before proceeding.")

    # Distinguish free enrollments from paid transactions.
    df["IsPaidTransaction"] = df["Amount"] > 0

    df["PaymentMethod"] = df["PaymentMethod"].astype("category")
    return df


def validate_referential_integrity(users, teachers, courses, transactions) -> None:
    """Raise if any foreign key in Transactions does not resolve. Verified
    clean in Phase 1, but re-checked here so the pipeline fails loudly if
    the source data ever changes."""
    bad_users = ~transactions["UserID"].isin(users["UserID"])
    bad_courses = ~transactions["CourseID"].isin(courses["CourseID"])
    bad_teachers = ~transactions["TeacherID"].isin(teachers["TeacherID"])

    if bad_users.any() or bad_courses.any() or bad_teachers.any():
        raise ValueError(
            "Referential integrity violation in Transactions: "
            f"{bad_users.sum()} bad UserID, {bad_courses.sum()} bad CourseID, "
            f"{bad_teachers.sum()} bad TeacherID."
        )


def preprocess_all(raw: dict) -> dict:
    """Run all cleaning steps and return a dict of processed DataFrames."""
    users = clean_users(raw["users"])
    teachers = clean_teachers(raw["teachers"])
    courses = clean_courses(raw["courses"])
    transactions = clean_transactions(raw["transactions"])

    validate_referential_integrity(users, teachers, courses, transactions)

    return {
        "users": users,
        "teachers": teachers,
        "courses": courses,
        "transactions": transactions,
    }


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    processed = preprocess_all(raw)
    for name, df in processed.items():
        print(f"{name}: shape={df.shape}, dtypes=\n{df.dtypes}\n")

    # Persist processed snapshots separately from the raw file (CSV keeps
    # this environment dependency-free; no parquet engine is installed).
    data_processed_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
    )
    os.makedirs(data_processed_dir, exist_ok=True)
    for name, df in processed.items():
        df.to_csv(os.path.join(data_processed_dir, f"{name}_processed.csv"), index=False)
    print(f"Saved processed sheets to {data_processed_dir}")
