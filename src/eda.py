"""
eda.py
------
Generates the EDA charts (saved to reports/figures/) and prints the
numeric findings that back the EDA_Report.md narrative. No chart or
number here is fabricated - every figure is produced directly from the
processed dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print("saved:", path)


def run_eda(course_df: pd.DataFrame, transactions: pd.DataFrame):
    # ---- D. Category distribution ----
    plt.figure(figsize=(9, 5))
    order = course_df["CourseCategory"].value_counts().index
    sns.countplot(data=course_df, y="CourseCategory", order=order, color="#4C72B0")
    plt.title("Number of Courses by Category")
    plt.xlabel("Course count"); plt.ylabel("")
    savefig("D_category_distribution.png")

    # ---- E. Course type distribution ----
    plt.figure(figsize=(5, 4))
    sns.countplot(data=course_df, x="CourseType", color="#55A868")
    plt.title("Course Type Distribution (Free vs Paid)")
    savefig("E_coursetype_distribution.png")

    # ---- F. Course level distribution ----
    plt.figure(figsize=(5, 4))
    sns.countplot(data=course_df, x="CourseLevel", order=["Beginner", "Intermediate", "Advanced"], color="#C44E52")
    plt.title("Course Level Distribution")
    savefig("F_courselevel_distribution.png")

    # ---- G. Price distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["CoursePrice"], bins=20, color="#8172B2")
    plt.title("Course Price Distribution")
    plt.xlabel("Price ($)")
    savefig("G_price_distribution.png")

    # ---- H. Duration distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["CourseDuration"], bins=20, color="#937860")
    plt.title("Course Duration Distribution")
    plt.xlabel("Duration (hours)")
    savefig("H_duration_distribution.png")

    # ---- I. Course rating distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["CourseRating"], bins=20, color="#DA8BC3")
    plt.title("Course Rating Distribution")
    savefig("I_courserating_distribution.png")

    # ---- J. Teacher experience distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["AvgTeacherExperience"], bins=15, color="#8C8C8C")
    plt.title("Average Teacher Experience per Course (years)")
    savefig("J_teacher_experience_distribution.png")

    # ---- K. Teacher rating distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["AvgTeacherRating"], bins=15, color="#CCB974")
    plt.title("Average Teacher Rating per Course")
    savefig("K_teacher_rating_distribution.png")

    # ---- L. Enrollment distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["EnrollmentCount"], bins=15, color="#64B5CD")
    plt.title("Enrollment Count Distribution (per course)")
    savefig("L_enrollment_distribution.png")

    # ---- M. Revenue distribution ----
    plt.figure(figsize=(7, 4))
    sns.histplot(course_df["TotalRevenue"], bins=20, color="#4C72B0")
    plt.title("Total Revenue Distribution (per course)")
    plt.xlabel("Revenue ($)")
    savefig("M_revenue_distribution.png")

    # ---- N. Enrollment by category ----
    plt.figure(figsize=(9, 5))
    cat_enroll = course_df.groupby("CourseCategory")["EnrollmentCount"].sum().sort_values()
    cat_enroll.plot(kind="barh", color="#55A868")
    plt.title("Total Enrollment by Course Category")
    plt.xlabel("Total enrollment")
    savefig("N_enrollment_by_category.png")

    # ---- O. Revenue by category ----
    plt.figure(figsize=(9, 5))
    cat_rev = course_df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values()
    cat_rev.plot(kind="barh", color="#C44E52")
    plt.title("Total Revenue by Course Category")
    plt.xlabel("Total revenue ($)")
    savefig("O_revenue_by_category.png")

    # ---- P. Top 10 courses by enrollment ----
    plt.figure(figsize=(9, 5))
    top_enroll = course_df.nlargest(10, "EnrollmentCount")[["CourseName", "EnrollmentCount"]].set_index("CourseName")
    top_enroll["EnrollmentCount"].sort_values().plot(kind="barh", color="#8172B2")
    plt.title("Top 10 Courses by Enrollment")
    savefig("P_top10_enrollment.png")

    # ---- Q. Top 10 courses by revenue ----
    plt.figure(figsize=(9, 5))
    top_rev = course_df.nlargest(10, "TotalRevenue")[["CourseName", "TotalRevenue"]].set_index("CourseName")
    top_rev["TotalRevenue"].sort_values().plot(kind="barh", color="#937860")
    plt.title("Top 10 Courses by Revenue")
    savefig("Q_top10_revenue.png")

    # ---- R. Price vs enrollment ----
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=course_df, x="CoursePrice", y="EnrollmentCount", hue="CourseType")
    plt.title("Course Price vs Enrollment")
    savefig("R_price_vs_enrollment.png")

    # ---- S. Rating vs enrollment ----
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=course_df, x="CourseRating", y="EnrollmentCount", hue="CourseLevel")
    plt.title("Course Rating vs Enrollment")
    savefig("S_rating_vs_enrollment.png")

    # ---- T. Duration vs enrollment ----
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=course_df, x="CourseDuration", y="EnrollmentCount")
    plt.title("Course Duration vs Enrollment")
    savefig("T_duration_vs_enrollment.png")

    # ---- U. Teacher rating vs enrollment ----
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=course_df, x="AvgTeacherRating", y="EnrollmentCount")
    plt.title("Avg Teacher Rating (per course) vs Enrollment")
    savefig("U_teacherrating_vs_enrollment.png")

    # ---- V. Teacher experience vs enrollment ----
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=course_df, x="AvgTeacherExperience", y="EnrollmentCount")
    plt.title("Avg Teacher Experience (per course) vs Enrollment")
    savefig("V_teacherexperience_vs_enrollment.png")

    # ---- W. Monthly enrollment trend ----
    trans = transactions.copy()
    trans["YearMonth"] = trans["TransactionDate"].dt.to_period("M").astype(str)
    monthly_enroll = trans.groupby("YearMonth").size()
    plt.figure(figsize=(9, 4))
    monthly_enroll.plot(kind="line", marker="o", color="#4C72B0")
    plt.title("Monthly Enrollment Trend (2025)")
    plt.ylabel("Enrollments"); plt.xticks(rotation=45)
    savefig("W_monthly_enrollment_trend.png")

    # ---- X. Monthly revenue trend ----
    monthly_rev = trans.groupby("YearMonth")["Amount"].sum()
    plt.figure(figsize=(9, 4))
    monthly_rev.plot(kind="line", marker="o", color="#C44E52")
    plt.title("Monthly Revenue Trend (2025)")
    plt.ylabel("Revenue ($)"); plt.xticks(rotation=45)
    savefig("X_monthly_revenue_trend.png")

    # ---- Y. Category-wise revenue trends over time ----
    trans_cat = trans.merge(course_df[["CourseID", "CourseCategory"]], on="CourseID", how="left")
    cat_month_rev = trans_cat.groupby(["YearMonth", "CourseCategory"])["Amount"].sum().unstack(fill_value=0)
    plt.figure(figsize=(11, 6))
    top5_cats = course_df.groupby("CourseCategory")["TotalRevenue"].sum().nlargest(5).index
    cat_month_rev[top5_cats].plot(ax=plt.gca())
    plt.title("Monthly Revenue Trend by Category (Top 5 categories)")
    plt.ylabel("Revenue ($)"); plt.xticks(rotation=45)
    savefig("Y_category_revenue_trends.png")

    # ---- Z. Correlation heatmap ----
    numeric_cols = ["CoursePrice", "CourseDuration", "CourseRating", "EnrollmentCount",
                     "TotalRevenue", "AvgTeacherRating", "AvgTeacherExperience",
                     "DistinctTeacherCount", "RecencyDays"]
    corr = course_df[numeric_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Course-Level Numeric Features)")
    savefig("Z_correlation_heatmap.png")

    return corr


if __name__ == "__main__":
    from data_loader import load_raw_data
    from preprocessing import preprocess_all
    from feature_engineering import build_course_level_dataset, add_engineered_features

    raw = load_raw_data()
    processed = preprocess_all(raw)
    course_df = build_course_level_dataset(processed["courses"], processed["transactions"], processed["teachers"])
    course_df = add_engineered_features(course_df)

    corr = run_eda(course_df, processed["transactions"])

    print("\n=== KEY EDA NUMBERS ===")
    print("Total courses:", len(course_df))
    print("Total enrollments (all transactions):", len(processed["transactions"]))
    print("Total revenue: $%.2f" % processed["transactions"]["Amount"].sum())
    print("Free vs Paid course count:\n", course_df["CourseType"].value_counts())
    print("\nEnrollment stats:\n", course_df["EnrollmentCount"].describe())
    print("\nRevenue stats:\n", course_df["TotalRevenue"].describe())
    print("\nCorrelation of EnrollmentCount with numeric features:\n",
          corr["EnrollmentCount"].sort_values(ascending=False))
    print("\nCorrelation of TotalRevenue with numeric features:\n",
          corr["TotalRevenue"].sort_values(ascending=False))
    print("\nTop category by total enrollment:\n",
          course_df.groupby("CourseCategory")["EnrollmentCount"].sum().sort_values(ascending=False).head(3))
    print("\nTop category by total revenue:\n",
          course_df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values(ascending=False).head(3))
