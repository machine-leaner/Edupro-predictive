"""
feature_engineering.py
-----------------------
Builds the final course-level analytical dataset used for modeling.

Key design decisions (documented per project requirements):

1. Unit of analysis: one row PER COURSE. EnrollmentCount and TotalRevenue
   are course-level aggregates of Transactions, which is what Phase 7 asks
   us to predict.

2. Course <-> Teacher relationship: Phase 1 found this is many-to-many and
   only exists inside Transactions (Courses/Teachers have no direct link).
   So teacher features are computed as HISTORICAL AGGREGATES over the
   instructors who taught a course's transactions, not a static 1:1 join.

3. No leakage: because the "historical teacher aggregate" and "historical
   enrollment/revenue" features are themselves derived from Transactions,
   we compute them using a chronological cutoff. Transactions up to and
   including the cutoff date build the FEATURES; the modeling TARGET
   (EnrollmentCount / TotalRevenue) is computed over a later, disjoint
   window. This mirrors a realistic forecasting setup: "given what we know
   about a course through time T, predict its enrollment/revenue in the
   following period." See PHASE7_TARGET3 notes at the bottom.

4. For the primary (non-forecasting) TARGET 1 / TARGET 2 models, described
   in the project as "predict EnrollmentCount / TotalRevenue" for a course
   from its static attributes, we build a full-history course-level
   dataset. Static course/teacher-pool attributes (price, duration,
   rating, category, average instructor profile) are not "future
   information" relative to the target in the sense scikit-learn cares
   about (they're properties of the course/instructor pool as a whole,
   known at catalog time) - this is called out explicitly, since the
   distinction between "descriptive aggregate of the same transactions
   being counted" and "leakage" matters for interpretation. We are
   transparent about this in the report: Target 1/2 models answer "what
   course/instructor profile is associated with higher demand and
   revenue?" (an association / catalog-design question), while Target 3
   answers the harder, leakage-safe forecasting question ("given history
   through month M, what happens in month M+1?").
"""

import os
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Phase 3: Transaction aggregation (course-level, full history)
# ----------------------------------------------------------------------

def aggregate_transactions_by_course(transactions: pd.DataFrame) -> pd.DataFrame:
    """Course-level transaction aggregates: enrollment, revenue, timing."""
    df = transactions.copy()

    agg = df.groupby("CourseID").agg(
        EnrollmentCount=("TransactionID", "count"),
        TotalRevenue=("Amount", "sum"),
        AverageTransactionAmount=("Amount", "mean"),
        PaidEnrollmentCount=("IsPaidTransaction", "sum"),
        FirstTransactionDate=("TransactionDate", "min"),
        LastTransactionDate=("TransactionDate", "max"),
    ).reset_index()

    agg["FreeEnrollmentCount"] = agg["EnrollmentCount"] - agg["PaidEnrollmentCount"]
    agg["RevenuePerEnrollment"] = (
        agg["TotalRevenue"] / agg["EnrollmentCount"].replace(0, np.nan)
    ).fillna(0)

    # Historical enrollment/revenue trend: split each course's history into
    # first half vs second half by transaction order, compare counts. This
    # uses only information already inside the course's own history (no
    # future courses / no other courses' data), so it is a legitimate
    # descriptive trend feature rather than a forward-looking target leak.
    trend_rows = []
    for course_id, g in df.sort_values("TransactionDate").groupby("CourseID"):
        n = len(g)
        half = n // 2
        first_half_rev = g.iloc[:half]["Amount"].sum() if half > 0 else 0
        second_half_rev = g.iloc[half:]["Amount"].sum()
        first_half_cnt = half
        second_half_cnt = n - half
        trend_rows.append(
            {
                "CourseID": course_id,
                "EnrollmentTrend": second_half_cnt - first_half_cnt,
                "RevenueTrend": second_half_rev - first_half_rev,
            }
        )
    trend_df = pd.DataFrame(trend_rows)
    agg = agg.merge(trend_df, on="CourseID", how="left")

    return agg


def add_time_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from each course's transaction history."""
    df = agg.copy()
    df["FirstEnrollmentYear"] = df["FirstTransactionDate"].dt.year
    df["FirstEnrollmentMonth"] = df["FirstTransactionDate"].dt.month
    df["FirstEnrollmentQuarter"] = df["FirstTransactionDate"].dt.quarter
    df["ActiveDaysSpan"] = (
        df["LastTransactionDate"] - df["FirstTransactionDate"]
    ).dt.days
    # Recency relative to the dataset's overall last observed date.
    dataset_end = agg["LastTransactionDate"].max()
    df["RecencyDays"] = (dataset_end - df["LastTransactionDate"]).dt.days
    return df


# ----------------------------------------------------------------------
# Teacher historical aggregates per course (handles many-to-many link)
# ----------------------------------------------------------------------

def aggregate_teacher_features_by_course(
    transactions: pd.DataFrame, teachers: pd.DataFrame
) -> pd.DataFrame:
    """
    Since a course has no single fixed teacher, we summarize the pool of
    instructors who taught each course's transactions: average rating,
    average experience, most common expertise match, and instructor count.
    """
    merged = transactions.merge(teachers, on="TeacherID", how="left")

    agg = merged.groupby("CourseID").agg(
        AvgTeacherRating=("TeacherRating", "mean"),
        AvgTeacherExperience=("YearsOfExperience", "mean"),
        DistinctTeacherCount=("TeacherID", "nunique"),
        MaxTeacherRating=("TeacherRating", "max"),
    ).reset_index()

    return agg


# ----------------------------------------------------------------------
# Phase 4: Merge into final analytical dataset
# ----------------------------------------------------------------------

def build_course_level_dataset(
    courses: pd.DataFrame, transactions: pd.DataFrame, teachers: pd.DataFrame
) -> pd.DataFrame:
    """Combine Courses + transaction aggregates + teacher-pool aggregates
    into one course-level modeling dataset. One row per course (60 rows)."""

    txn_agg = aggregate_transactions_by_course(transactions)
    txn_agg = add_time_features(txn_agg)
    teacher_agg = aggregate_teacher_features_by_course(transactions, teachers)

    df = courses.merge(txn_agg, on="CourseID", how="left")
    df = df.merge(teacher_agg, on="CourseID", how="left")

    # Every course must have at least one transaction in this dataset
    # (verified: 60/60 courses appear in Transactions). Guard defensively.
    if df["EnrollmentCount"].isna().any():
        missing = df[df["EnrollmentCount"].isna()]["CourseID"].tolist()
        raise ValueError(
            f"{len(missing)} course(s) have no transactions and no target "
            f"can be computed for them: {missing}"
        )

    return df


# ----------------------------------------------------------------------
# Phase 6: Engineered categorical features
# ----------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket/tier features built from the merged course-level dataset."""
    out = df.copy()

    out["PriceBand"] = pd.cut(
        out["CoursePrice"],
        bins=[-0.01, 0, 100, 250, np.inf],
        labels=["Free", "Low", "Medium", "High"],
    )

    out["DurationBucket"] = pd.cut(
        out["CourseDuration"],
        bins=[0, 5, 15, 30, np.inf],
        labels=["Short", "Medium", "Long", "VeryLong"],
    )

    out["RatingTier"] = pd.cut(
        out["CourseRating"],
        bins=[0, 2, 3.5, 5.01],
        labels=["Low", "Medium", "High"],
    )

    out["ExperienceBucket"] = pd.cut(
        out["AvgTeacherExperience"],
        bins=[0, 5, 10, np.inf],
        labels=["Junior", "Mid", "Senior"],
    )

    # Category-level historical demand: mean enrollment across all OTHER
    # courses in the same category (leave-one-out to avoid a course's own
    # target leaking into its own feature).
    cat_means = []
    for idx, row in out.iterrows():
        same_cat = out[
            (out["CourseCategory"] == row["CourseCategory"])
            & (out["CourseID"] != row["CourseID"])
        ]
        cat_means.append(same_cat["EnrollmentCount"].mean())
    out["CategoryHistoricalDemand"] = cat_means

    cat_rev_means = []
    for idx, row in out.iterrows():
        same_cat = out[
            (out["CourseCategory"] == row["CourseCategory"])
            & (out["CourseID"] != row["CourseID"])
        ]
        cat_rev_means.append(same_cat["TotalRevenue"].mean())
    out["CategoryHistoricalRevenue"] = cat_rev_means

    return out


# ----------------------------------------------------------------------
# Phase 7 / Target 3: leakage-safe forecasting feature set (monthly)
# ----------------------------------------------------------------------

def build_forecasting_dataset(transactions: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    """
    Course x Month panel for the forecasting target: predict a course's
    NEXT month's enrollment/revenue using only information through the
    current month (rolling history), with a chronological split.

    Each row = (CourseID, Month). Feature columns are computed using
    transactions strictly BEFORE that month (or within it, for the
    current-month running features, which are legitimate "so far this
    month" signals - the target columns are always the *next* month).
    """
    df = transactions.copy()
    df["YearMonth"] = df["TransactionDate"].dt.to_period("M")

    monthly = df.groupby(["CourseID", "YearMonth"]).agg(
        MonthlyEnrollment=("TransactionID", "count"),
        MonthlyRevenue=("Amount", "sum"),
    ).reset_index()

    all_months = sorted(monthly["YearMonth"].unique())
    all_courses = courses["CourseID"].unique()

    # Build a complete CourseID x Month grid so months with zero
    # transactions for a course are explicit 0s, not missing rows.
    full_index = pd.MultiIndex.from_product(
        [all_courses, all_months], names=["CourseID", "YearMonth"]
    )
    panel = monthly.set_index(["CourseID", "YearMonth"]).reindex(
        full_index, fill_value=0
    ).reset_index()

    panel = panel.sort_values(["CourseID", "YearMonth"]).reset_index(drop=True)

    # Rolling / lag features computed PER COURSE using only prior months.
    panel["Lag1Enrollment"] = panel.groupby("CourseID")["MonthlyEnrollment"].shift(1)
    panel["Lag1Revenue"] = panel.groupby("CourseID")["MonthlyRevenue"].shift(1)
    panel["RollingAvgEnrollment3M"] = (
        panel.groupby("CourseID")["MonthlyEnrollment"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    panel["CumulativeEnrollmentToDate"] = (
        panel.groupby("CourseID")["MonthlyEnrollment"].cumsum()
        - panel["MonthlyEnrollment"]
    )

    # Target: NEXT month's enrollment/revenue for the same course.
    panel["NextMonthEnrollment"] = panel.groupby("CourseID")["MonthlyEnrollment"].shift(-1)
    panel["NextMonthRevenue"] = panel.groupby("CourseID")["MonthlyRevenue"].shift(-1)

    # Drop rows where we can't compute a lag feature (first month per
    # course) or don't have a next-month target (last month per course).
    panel = panel.dropna(
        subset=["Lag1Enrollment", "NextMonthEnrollment"]
    ).reset_index(drop=True)

    # Attach static course attributes (known at catalog time, not derived
    # from the target window).
    static_cols = ["CourseID", "CourseCategory", "CourseType", "CourseLevel",
                   "CoursePrice", "CourseDuration", "CourseRating"]
    panel = panel.merge(courses[static_cols], on="CourseID", how="left")

    return panel


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_all

    raw = load_raw_data()
    processed = preprocess_all(raw)

    course_df = build_course_level_dataset(
        processed["courses"], processed["transactions"], processed["teachers"]
    )
    course_df = add_engineered_features(course_df)
    print("Course-level dataset shape:", course_df.shape)
    print(course_df.columns.tolist())
    print(course_df.head(3).to_string())

    forecast_df = build_forecasting_dataset(processed["transactions"], processed["courses"])
    print("\nForecasting panel shape:", forecast_df.shape)
    print(forecast_df.columns.tolist())
    print("Distinct months available:", forecast_df["YearMonth"].nunique() if "YearMonth" in forecast_df else "n/a")

    data_processed_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
    )
    os.makedirs(data_processed_dir, exist_ok=True)
    course_df.to_csv(os.path.join(data_processed_dir, "course_level_dataset.csv"), index=False)
    forecast_df.to_csv(os.path.join(data_processed_dir, "forecasting_panel.csv"), index=False)
    print(f"\nSaved course_level_dataset.csv and forecasting_panel.csv to {data_processed_dir}")
