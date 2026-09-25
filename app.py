"""
EduPro Predictive Analytics Dashboard
--------------------------------------
Streamlit app for course demand and revenue analytics/prediction.

Run with:  streamlit run app.py

Loads pre-computed processed data and pre-trained models (see src/) rather
than retraining on every page load.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import generic_analyzer as ga

st.set_page_config(page_title="EduPro Predictive Analytics", layout="wide", page_icon="\U0001F4CA")


# ----------------------------------------------------------------------
# Cached data / model loaders
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    course_df = pd.read_csv(os.path.join(DATA_DIR, "course_level_dataset.csv"),
                             parse_dates=["FirstTransactionDate", "LastTransactionDate"])
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions_processed.csv"),
                                parse_dates=["TransactionDate"])
    users = pd.read_csv(os.path.join(DATA_DIR, "users_processed.csv"))
    teachers = pd.read_csv(os.path.join(DATA_DIR, "teachers_processed.csv"))
    forecast_panel = pd.read_csv(os.path.join(DATA_DIR, "forecasting_panel.csv"))
    return course_df, transactions, users, teachers, forecast_panel


@st.cache_resource
def load_models():
    demand_model = joblib.load(os.path.join(MODELS_DIR, "demand_model.pkl"))
    revenue_model = joblib.load(os.path.join(MODELS_DIR, "revenue_model.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return demand_model, revenue_model, meta


@st.cache_data
def load_reports():
    reports = {}
    for name in ["enrollment_model_comparison", "revenue_model_comparison",
                 "forecast_model_comparison", "enrollment_feature_importance",
                 "revenue_feature_importance"]:
        path = os.path.join(REPORTS_DIR, f"{name}.csv")
        if os.path.exists(path):
            reports[name] = pd.read_csv(path)
    return reports


course_df, transactions, users, teachers, forecast_panel = load_data()
demand_model, revenue_model, meta = load_models()
reports = load_reports()

st.sidebar.title("\U0001F4DA EduPro Analytics")
st.sidebar.caption("Navigation")
page = st.sidebar.radio(
    "Navigate",
    [
        "\U0001F4CA Executive Dashboard",
        "\U0001F4DD Dataset Overview",
        "\U0001F4DA Course Demand Analysis",
        "\U0001F4B0 Revenue Analysis",
        "\U0001F916 Demand Prediction",
        "\U0001F4C8 Revenue Prediction",
        "\U0001F50D Category Analysis",
        "\u2699\uFE0F Feature Importance",
        "\U0001F4A1 Business Recommendations",
        "\U0001F4C1 Upload & Analyze Dataset",
    ],
    label_visibility="collapsed",
)
# Strip the leading emoji so the rest of the file's `page == "..."` checks
# (written before icons were added) keep working unchanged.
page = page.split(" ", 1)[1] if " " in page else page

CATEGORY_OPTIONS = sorted(course_df["CourseCategory"].unique())
TYPE_OPTIONS = sorted(course_df["CourseType"].unique())
LEVEL_OPTIONS = sorted(course_df["CourseLevel"].unique())


# ----------------------------------------------------------------------
# 1. Executive Dashboard
# ----------------------------------------------------------------------
if page == "Executive Dashboard":
    st.title("Executive Dashboard")
    st.caption("EduPro Online Platform \u2014 course demand & revenue at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Courses", f"{course_df['CourseID'].nunique():,}")
    c2.metric("Total Teachers", f"{teachers['TeacherID'].nunique():,}")
    c3.metric("Total Users", f"{users['UserID'].nunique():,}")
    c4.metric("Total Transactions", f"{len(transactions):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Enrollments", f"{course_df['EnrollmentCount'].sum():,}")
    c6.metric("Total Revenue", f"${course_df['TotalRevenue'].sum():,.2f}")
    c7.metric("Avg Course Rating", f"{course_df['CourseRating'].mean():.2f} / 5")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        trans_m = transactions.copy()
        trans_m["Month"] = trans_m["TransactionDate"].dt.to_period("M").astype(str)
        monthly = trans_m.groupby("Month").size().reset_index(name="Enrollments")
        fig = px.line(monthly, x="Month", y="Enrollments", markers=True,
                      title="Monthly Enrollment Trend")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        monthly_rev = trans_m.groupby("Month")["Amount"].sum().reset_index(name="Revenue")
        fig = px.line(monthly_rev, x="Month", y="Revenue", markers=True,
                      title="Monthly Revenue Trend", color_discrete_sequence=["#C44E52"])
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cat_enroll = course_df.groupby("CourseCategory")["EnrollmentCount"].sum().sort_values().reset_index()
        fig = px.bar(cat_enroll, x="EnrollmentCount", y="CourseCategory", orientation="h",
                     title="Enrollment by Category")
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        cat_rev = course_df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values().reset_index()
        fig = px.bar(cat_rev, x="TotalRevenue", y="CourseCategory", orientation="h",
                     title="Revenue by Category", color_discrete_sequence=["#C44E52"])
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# 2. Dataset Overview
# ----------------------------------------------------------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")

    st.subheader("Source Tables")
    t1, t2, t3, t4 = st.tabs(["Users", "Teachers", "Courses", "Transactions"])
    with t1:
        st.write(f"Shape: {users.shape}")
        st.dataframe(users.head(20), use_container_width=True)
    with t2:
        st.write(f"Shape: {teachers.shape}")
        st.dataframe(teachers.head(20), use_container_width=True)
    with t3:
        st.write(f"Shape: {course_df[['CourseID','CourseName','CourseCategory','CourseType','CourseLevel','CoursePrice','CourseDuration','CourseRating']].shape}")
        st.dataframe(course_df[['CourseID','CourseName','CourseCategory','CourseType','CourseLevel','CoursePrice','CourseDuration','CourseRating']], use_container_width=True)
    with t4:
        st.write(f"Shape: {transactions.shape}")
        st.dataframe(transactions.head(20), use_container_width=True)

    st.divider()
    st.subheader("Data Quality Summary")
    st.markdown("""
    - **No missing values and no duplicate rows** were found in any of the four sheets.
    - **All foreign keys resolve**: every UserID, CourseID, and TeacherID referenced in
      Transactions exists in its parent sheet.
    - **Course \u2194 Teacher relationship is many-to-many**, present only inside Transactions
      (Courses/Teachers have no direct link). Teacher features used in modeling are
      therefore historical aggregates over the instructor pool per course, not a fixed join.
    - **Transaction Amount always exactly equals the course's CoursePrice** (0 exceptions) \u2014
      there is no variation from discounts/refunds in this dataset, so **TotalRevenue is a
      deterministic function of CoursePrice \u00d7 EnrollmentCount.**
    - 38 of 60 courses are free (CoursePrice = 0), which shows up as 6,403 of 10,000
      Amount = 0 transactions.
    """)

    st.subheader("Final Course-Level Modeling Dataset")
    st.write(f"Shape: {course_df.shape}")
    st.dataframe(course_df.describe(include="all").transpose(), use_container_width=True)


# ----------------------------------------------------------------------
# 3. Course Demand Analysis
# ----------------------------------------------------------------------
elif page == "Course Demand Analysis":
    st.title("Course Demand Analysis")

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        sel_cat = f1.multiselect("Category", CATEGORY_OPTIONS, default=CATEGORY_OPTIONS)
        sel_type = f2.multiselect("Course Type", TYPE_OPTIONS, default=TYPE_OPTIONS)
        sel_level = f3.multiselect("Course Level", LEVEL_OPTIONS, default=LEVEL_OPTIONS)
        date_range = st.date_input(
            "Transaction date range",
            value=(transactions["TransactionDate"].min().date(), transactions["TransactionDate"].max().date()),
        )

    filtered = course_df[
        course_df["CourseCategory"].isin(sel_cat)
        & course_df["CourseType"].isin(sel_type)
        & course_df["CourseLevel"].isin(sel_level)
    ]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        trans_filtered = transactions[
            (transactions["TransactionDate"] >= start) & (transactions["TransactionDate"] <= end)
            & transactions["CourseID"].isin(filtered["CourseID"])
        ]
    else:
        trans_filtered = transactions[transactions["CourseID"].isin(filtered["CourseID"])]

    c1, c2 = st.columns(2)
    with c1:
        cat_enroll = filtered.groupby("CourseCategory")["EnrollmentCount"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(cat_enroll, x="EnrollmentCount", y="CourseCategory", orientation="h",
                                title="Enrollment by Category"), use_container_width=True)
    with c2:
        top10 = filtered.nlargest(10, "EnrollmentCount")[["CourseName", "EnrollmentCount"]]
        st.plotly_chart(px.bar(top10.sort_values("EnrollmentCount"), x="EnrollmentCount", y="CourseName",
                                orientation="h", title="Top 10 Courses by Enrollment"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        level_demand = filtered.groupby("CourseLevel")["EnrollmentCount"].sum().reset_index()
        st.plotly_chart(px.bar(level_demand, x="CourseLevel", y="EnrollmentCount",
                                title="Demand by Course Level"), use_container_width=True)
    with c4:
        type_demand = filtered.groupby("CourseType")["EnrollmentCount"].sum().reset_index()
        st.plotly_chart(px.bar(type_demand, x="CourseType", y="EnrollmentCount",
                                title="Demand by Course Type", color_discrete_sequence=["#55A868"]),
                         use_container_width=True)

    trans_m = trans_filtered.copy()
    trans_m["Month"] = trans_m["TransactionDate"].dt.to_period("M").astype(str)
    monthly = trans_m.groupby("Month").size().reset_index(name="Enrollments")
    st.plotly_chart(px.line(monthly, x="Month", y="Enrollments", markers=True,
                             title="Monthly Enrollment Trend (filtered)"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(px.scatter(filtered, x="CoursePrice", y="EnrollmentCount", color="CourseType",
                                    hover_data=["CourseName"], title="Price vs Enrollment"),
                         use_container_width=True)
    with c6:
        st.plotly_chart(px.scatter(filtered, x="CourseRating", y="EnrollmentCount", color="CourseLevel",
                                    hover_data=["CourseName"], title="Rating vs Enrollment"),
                         use_container_width=True)

    st.caption("Correlation between Course Rating and Enrollment across all 60 courses is "
               "weakly positive (~0.29); Price shows a weak negative correlation (~-0.16). "
               "These are associations only, not causal claims.")


# ----------------------------------------------------------------------
# 4. Revenue Analysis
# ----------------------------------------------------------------------
elif page == "Revenue Analysis":
    st.title("Revenue Analysis")

    st.info("Amount in this dataset always exactly equals CoursePrice, so TotalRevenue = "
            "CoursePrice \u00d7 EnrollmentCount deterministically \u2014 revenue trends here mirror "
            "price and enrollment patterns rather than an independent signal.")

    c1, c2 = st.columns(2)
    c1.metric("Total Revenue", f"${course_df['TotalRevenue'].sum():,.2f}")
    c2.metric("Avg Revenue per Course", f"${course_df['TotalRevenue'].mean():,.2f}")

    c3, c4 = st.columns(2)
    with c3:
        cat_rev = course_df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(cat_rev, x="TotalRevenue", y="CourseCategory", orientation="h",
                                title="Revenue by Category"), use_container_width=True)
    with c4:
        top10_rev = course_df.nlargest(10, "TotalRevenue")[["CourseName", "TotalRevenue"]]
        st.plotly_chart(px.bar(top10_rev.sort_values("TotalRevenue"), x="TotalRevenue", y="CourseName",
                                orientation="h", title="Top 10 Courses by Revenue",
                                color_discrete_sequence=["#C44E52"]), use_container_width=True)

    trans_m = transactions.copy()
    trans_m["Month"] = trans_m["TransactionDate"].dt.to_period("M").astype(str)
    monthly_rev = trans_m.groupby("Month")["Amount"].sum().reset_index(name="Revenue")
    st.plotly_chart(px.line(monthly_rev, x="Month", y="Revenue", markers=True,
                             title="Monthly Revenue Trend", color_discrete_sequence=["#C44E52"]),
                     use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(px.scatter(course_df, x="CoursePrice", y="TotalRevenue", color="CourseCategory",
                                    hover_data=["CourseName"], title="Revenue vs Price"),
                         use_container_width=True)
    with c6:
        st.plotly_chart(px.scatter(course_df, x="EnrollmentCount", y="TotalRevenue", color="CourseCategory",
                                    hover_data=["CourseName"], title="Revenue vs Enrollment"),
                         use_container_width=True)


# ----------------------------------------------------------------------
# 5. Demand Prediction
# ----------------------------------------------------------------------
elif page == "Demand Prediction":
    st.title("Demand Prediction")
    st.caption(f"Model: {meta['enrollment_model']} (selected by lowest LOOCV MAE)")

    st.warning("This model's out-of-sample R\u00b2 is near zero (Leave-One-Out cross-validation "
               "on 60 courses) \u2014 course/teacher attributes in this dataset show very little "
               "predictive signal for enrollment. Treat predictions as a rough, low-confidence "
               "estimate, not a reliable forecast.")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Course Category", CATEGORY_OPTIONS)
        course_type = st.selectbox("Course Type", TYPE_OPTIONS)
        level = st.selectbox("Course Level", LEVEL_OPTIONS)
        price = st.number_input("Course Price ($)", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
    with col2:
        duration = st.number_input("Course Duration (hours)", min_value=0.5, max_value=100.0, value=15.0, step=0.5)
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, 0.1)
        teacher_experience = st.slider("Avg Teacher Experience (years)", 1, 25, 8)
        teacher_rating = st.slider("Avg Teacher Rating", 1.0, 5.0, 4.0, 0.1)
    distinct_teachers = st.slider("Distinct Teachers Expected", 1, 30, 10)

    if st.button("Predict Enrollment", type="primary"):
        inputs = {
            "CoursePrice": price, "CourseDuration": duration, "CourseRating": rating,
            "AvgTeacherRating": teacher_rating, "AvgTeacherExperience": teacher_experience,
            "DistinctTeacherCount": distinct_teachers,
            "CourseCategory": category, "CourseType": course_type, "CourseLevel": level,
        }
        cols = meta["enrollment_features_numeric"] + meta["enrollment_features_categorical"]
        row = pd.DataFrame([{c: inputs[c] for c in cols}])
        pred = demand_model.predict(row)[0]
        st.success(f"### Predicted Enrollment: {pred:,.0f} students")

        st.markdown("**Major factors influencing enrollment predictions in this model** "
                    "(from RandomForest feature importance on the same data): CourseRating and "
                    "CourseDuration rank highest, followed by the size/experience of the "
                    "instructor pool. These are model associations, not proven causes.")


# ----------------------------------------------------------------------
# 6. Revenue Prediction
# ----------------------------------------------------------------------
elif page == "Revenue Prediction":
    st.title("Revenue Prediction")
    st.caption(f"Model: {meta['revenue_model']} (selected by lowest LOOCV MAE)")
    st.info("Predictions are estimates based on historical patterns, not guarantees.")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Course Category", CATEGORY_OPTIONS, key="rev_cat")
        course_type = st.selectbox("Course Type", TYPE_OPTIONS, key="rev_type")
        level = st.selectbox("Course Level", LEVEL_OPTIONS, key="rev_level")
        price = st.number_input("Course Price ($)", min_value=0.0, max_value=1000.0, value=150.0, step=10.0, key="rev_price")
    with col2:
        duration = st.number_input("Course Duration (hours)", min_value=0.5, max_value=100.0, value=15.0, step=0.5, key="rev_dur")
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, 0.1, key="rev_rating")
        teacher_experience = st.slider("Avg Teacher Experience (years)", 1, 25, 8, key="rev_exp")
        teacher_rating = st.slider("Avg Teacher Rating", 1.0, 5.0, 4.0, 0.1, key="rev_teacher_rating")
    distinct_teachers = st.slider("Distinct Teachers Expected", 1, 30, 10, key="rev_distinct")

    if st.button("Predict Revenue", type="primary"):
        inputs = {
            "CoursePrice": price, "CourseDuration": duration, "CourseRating": rating,
            "AvgTeacherRating": teacher_rating, "AvgTeacherExperience": teacher_experience,
            "DistinctTeacherCount": distinct_teachers,
            "CourseCategory": category, "CourseType": course_type, "CourseLevel": level,
        }
        cols = meta["revenue_features_numeric"] + meta["revenue_features_categorical"]
        row = pd.DataFrame([{c: inputs[c] for c in cols}])
        pred_rev = revenue_model.predict(row)[0]

        # Also show implied demand model prediction for context.
        demand_cols = meta["enrollment_features_numeric"] + meta["enrollment_features_categorical"]
        demand_row = pd.DataFrame([{c: inputs[c] for c in demand_cols}])
        pred_enroll = demand_model.predict(demand_row)[0]

        st.success(f"### Predicted Revenue (Estimate): ${pred_rev:,.2f}")
        c1, c2 = st.columns(2)
        c1.metric("Category", category)
        c2.metric("Expected Demand (from Demand model)", f"{pred_enroll:,.0f} students")
        st.caption("Because Amount always equals CoursePrice in this dataset, revenue is "
                    "almost entirely driven by the price you set \u00d7 predicted enrollment \u2014 "
                    "CoursePrice alone explains ~97% of the RandomForest's revenue importance.")


# ----------------------------------------------------------------------
# 7. Category Analysis
# ----------------------------------------------------------------------
elif page == "Category Analysis":
    st.title("Category Analysis")

    cat_summary = course_df.groupby("CourseCategory").agg(
        Courses=("CourseID", "count"),
        TotalEnrollment=("EnrollmentCount", "sum"),
        TotalRevenue=("TotalRevenue", "sum"),
        AvgPrice=("CoursePrice", "mean"),
        AvgRating=("CourseRating", "mean"),
    ).sort_values("TotalRevenue", ascending=False).reset_index()

    st.dataframe(cat_summary.style.format({
        "TotalRevenue": "${:,.2f}", "AvgPrice": "${:,.2f}", "AvgRating": "{:.2f}"
    }), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(cat_summary, x="CourseCategory", y="TotalEnrollment",
                                title="Total Enrollment by Category"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(cat_summary, x="CourseCategory", y="TotalRevenue",
                                title="Total Revenue by Category", color_discrete_sequence=["#C44E52"]),
                         use_container_width=True)

    trans_cat = transactions.merge(course_df[["CourseID", "CourseCategory"]], on="CourseID", how="left")
    trans_cat["Month"] = trans_cat["TransactionDate"].dt.to_period("M").astype(str)
    top5 = cat_summary.nlargest(5, "TotalRevenue")["CourseCategory"].tolist()
    cat_month = trans_cat[trans_cat["CourseCategory"].isin(top5)].groupby(
        ["Month", "CourseCategory"])["Amount"].sum().reset_index()
    st.plotly_chart(px.line(cat_month, x="Month", y="Amount", color="CourseCategory",
                             title="Monthly Revenue Trend \u2014 Top 5 Categories"), use_container_width=True)


# ----------------------------------------------------------------------
# 8. Feature Importance
# ----------------------------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance / Explainability")
    st.caption("RandomForest impurity-based importance \u2014 reflects predictive association "
               "within the model, not proof of causation.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enrollment (Demand) Prediction")
        if "enrollment_feature_importance" in reports:
            imp = reports["enrollment_feature_importance"].head(12).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h"),
                             use_container_width=True)
        st.caption("Note: the enrollment model's LOOCV R\u00b2 is near zero, so these rankings "
                   "should be read as weak, exploratory signals only.")
    with col2:
        st.subheader("Revenue Prediction")
        if "revenue_feature_importance" in reports:
            imp = reports["revenue_feature_importance"].head(12).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                                    color_discrete_sequence=["#C44E52"]), use_container_width=True)
        st.caption("CoursePrice dominates revenue importance (~97%) because Amount always "
                   "equals CoursePrice in this dataset.")

    st.divider()
    st.subheader("Model Comparison Tables (LOOCV, n=60 courses)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Enrollment models**")
        if "enrollment_model_comparison" in reports:
            st.dataframe(reports["enrollment_model_comparison"], use_container_width=True)
    with c2:
        st.markdown("**Revenue models**")
        if "revenue_model_comparison" in reports:
            st.dataframe(reports["revenue_model_comparison"], use_container_width=True)

    st.markdown("**Forecasting models (chronological split, next-month enrollment)**")
    if "forecast_model_comparison" in reports:
        st.dataframe(reports["forecast_model_comparison"], use_container_width=True)


# ----------------------------------------------------------------------
# 9. Business Recommendations
# ----------------------------------------------------------------------
elif page == "Business Recommendations":
    st.title("Business Recommendations")

    top_enroll_cat = course_df.groupby("CourseCategory")["EnrollmentCount"].sum().idxmax()
    top_rev_cat = course_df.groupby("CourseCategory")["TotalRevenue"].sum().idxmax()
    price_corr = course_df["CoursePrice"].corr(course_df["EnrollmentCount"])
    rating_corr = course_df["CourseRating"].corr(course_df["EnrollmentCount"])

    st.markdown(f"""
    **1. Strongest observed demand:** *{top_enroll_cat}* leads all categories in total
    historical enrollment.

    **2. Strongest observed revenue:** *{top_rev_cat}* generates the highest total revenue,
    driven primarily by its pricing rather than a demand advantage (recall Amount =
    CoursePrice \u00d7 EnrollmentCount deterministically in this dataset).

    **3. Price vs. demand:** the correlation between CoursePrice and EnrollmentCount across
    all 60 courses is **{price_corr:.2f}** \u2014 a weak negative association. Price alone is not
    a strong predictor of course popularity here.

    **4. Rating vs. demand:** the correlation between CourseRating and EnrollmentCount is
    **{rating_corr:.2f}** \u2014 a weak positive association, the strongest single numeric
    association found for demand, but still not strong enough to build a confident
    forecasting model (LOOCV R\u00b2 \u2248 0).

    **5. Teacher effects:** average instructor rating/experience per course show negligible
    correlation with enrollment in this dataset (see Feature Importance page).

    **6. Most influential model features:** CourseRating and CourseDuration rank highest for
    enrollment; CoursePrice overwhelmingly dominates revenue (a mechanical consequence of the
    data, not a business insight about pricing elasticity).

    **7 & 8. Categories to watch:** {top_enroll_cat} (demand) and {top_rev_cat} (revenue) are
    the categories with the strongest historical track record and are reasonable starting
    points for new course investment, though the weak overall model signal means this should
    be paired with qualitative market research before committing budget.

    **9. Seasonality:** monthly enrollment volume across 2025 ranges from about 762 to 899
    transactions/month \u2014 mild month-to-month variation with no dramatic seasonal spike.

    **10. Pricing:** because revenue is a deterministic function of price, EduPro can compute
    exact revenue impact of a price change once expected enrollment is estimated \u2014 but this
    dataset does not show strong evidence that lower prices meaningfully increase enrollment
    (price/enrollment correlation \u2248 {price_corr:.2f}), so blanket discounting is not
    obviously supported by the data.
    """)

    st.warning("**Honesty note:** the enrollment (demand) models in this project show "
               "near-zero out-of-sample explanatory power (LOOCV R\u00b2 \u2248 0 for all algorithms "
               "tried). This is a genuine finding, not a modeling failure to hide \u2014 with only "
               "60 courses and largely price-independent enrollment counts, this dataset does "
               "not contain a strong learnable demand signal from catalog attributes alone. "
               "Recommendations above should be read as descriptive observations from the "
               "historical data, not as validated causal drivers of demand.")


# ----------------------------------------------------------------------
# 10. Upload & Analyze Dataset (Generic Dataset Analyzer - Mode 2)
# ----------------------------------------------------------------------
elif page == "Upload & Analyze Dataset":
    st.title("Upload & Analyze Dataset")
    st.caption("Bring your own CSV or Excel file for automatic profiling, EDA, and ML modeling "
               "\u2014 this mode is completely independent of the EduPro dashboard above.")

    # ---- Session-state defaults (persist the workflow across tabs/reruns) ----
    for key, default in [
        ("gen_df", None), ("gen_filename", None), ("gen_sheet", None),
        ("gen_col_types", None), ("gen_working_df", None),
        ("gen_target", None), ("gen_problem_type", None),
        ("gen_results_df", None), ("gen_pipelines", None),
        ("gen_X_test", None), ("gen_y_test", None), ("gen_split", None),
        ("gen_best_model_name", None), ("gen_importance_df", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.subheader("1. Upload Your Dataset")
    uploaded = st.file_uploader(
        "Drag and drop a CSV or Excel file here, or click Browse files",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded is not None:
        is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
        new_file = uploaded.name != st.session_state["gen_filename"]

        sheet_to_load = None
        if is_excel:
            try:
                sheet_names = ga.get_excel_sheet_names(uploaded.getvalue())
            except (ga.UnsupportedFileError, ga.EmptyDatasetError) as e:
                st.error(f"\u26A0\uFE0F {e}")
                sheet_names = []
            if sheet_names:
                if len(sheet_names) > 1:
                    st.info(f"This workbook has {len(sheet_names)} sheets.")
                    sheet_to_load = st.selectbox("Select a sheet to analyze", sheet_names, key="gen_sheet_select")
                else:
                    sheet_to_load = sheet_names[0]

        should_load = new_file or (is_excel and sheet_to_load != st.session_state["gen_sheet"])

        if should_load and (not is_excel or sheet_to_load is not None):
            try:
                df = ga.load_uploaded_file(uploaded, sheet_name=sheet_to_load)
                st.session_state["gen_df"] = df
                st.session_state["gen_filename"] = uploaded.name
                st.session_state["gen_sheet"] = sheet_to_load
                st.session_state["gen_col_types"] = ga.detect_column_types(df)
                st.session_state["gen_working_df"] = df.copy()
                # Reset downstream state on a genuinely new dataset/sheet.
                st.session_state["gen_target"] = None
                st.session_state["gen_results_df"] = None
                st.session_state["gen_pipelines"] = None
                st.session_state["gen_importance_df"] = None
                st.session_state["gen_last_target_for_suggestion"] = None
                # Also clear stale widget state that references column names
                # from the PREVIOUS dataset (e.g. a target/outlier/missing
                # column selectbox pointing at a column that no longer
                # exists), so widgets re-initialize cleanly for the new data.
                for widget_key in ["gen_target_select", "gen_problem_type_select",
                                    "gen_outlier_col", "gen_outlier_action",
                                    "gen_miss_col", "gen_miss_strategy"]:
                    st.session_state.pop(widget_key, None)
                st.success(f"Loaded **{uploaded.name}**"
                           + (f" (sheet: {sheet_to_load})" if sheet_to_load else "") + ".")
            except (ga.UnsupportedFileError, ga.EmptyDatasetError) as e:
                st.error(f"\u26A0\uFE0F {e}")
            except Exception as e:
                st.error("\u26A0\uFE0F Something went wrong while reading this file. "
                          "Please check the file is a valid, well-formed CSV/Excel file.")

    df = st.session_state["gen_working_df"]

    if df is None:
        st.info("Upload a CSV or Excel file above to begin automatic analysis.")
    else:
        col_types = ga.detect_column_types(df)
        st.session_state["gen_col_types"] = col_types

        if len(df) > 200_000:
            st.warning(f"This dataset has {len(df):,} rows \u2014 charts below will sample "
                       "a subset for responsiveness.")

        tab_overview, tab_quality, tab_eda, tab_prep, tab_model, tab_insights = st.tabs(
            ["Overview", "Data Quality", "EDA", "Preprocessing", "Modeling", "Insights"]
        )

        # ---------------- Overview ----------------
        with tab_overview:
            info = ga.get_basic_info(df)
            st.markdown(f"**Dataset:** {st.session_state['gen_filename']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{info['n_rows']:,}")
            c2.metric("Columns", f"{info['n_columns']:,}")
            c3.metric("Memory Usage", f"{info['memory_usage_mb']} MB")

            n_show = min(10, len(df))
            st.markdown(f"**First {n_show} rows**")
            st.dataframe(df.head(n_show), use_container_width=True)

            st.markdown("**Column names & data types**")
            st.dataframe(
                pd.DataFrame({"Column": info["columns"],
                              "Data Type": [info["dtypes"][c] for c in info["columns"]]}),
                use_container_width=True,
            )

        # ---------------- Data Quality ----------------
        with tab_quality:
            summary = ga.data_quality_summary(df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{summary['Rows']:,}")
            c2.metric("Columns", f"{summary['Columns']:,}")
            c3.metric("Missing Values", f"{summary['Missing Values']:,}")
            c4.metric("Duplicate Rows", f"{summary['Duplicate Rows']:,}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Numerical Columns", len(col_types["numeric_cols"]))
            c6.metric("Categorical Columns", len(col_types["categorical_cols"]))
            c7.metric("Date Columns", len(col_types["date_cols"]))
            c8.metric("Potential ID Columns", len(col_types["id_cols"]))

            if col_types["constant_cols"]:
                st.caption(f"Constant columns (single value, excluded from modeling): "
                           f"{', '.join(col_types['constant_cols'])}")

            st.markdown("**Column-level data quality**")
            st.dataframe(ga.data_quality_table(df), use_container_width=True)

            st.markdown("**Detected column groups**")
            g1, g2, g3, g4 = st.columns(4)
            g1.markdown("**Numerical**\n" + "\n".join(f"- {c}" for c in col_types["numeric_cols"]) or "_none_")
            g2.markdown("**Categorical**\n" + "\n".join(f"- {c}" for c in col_types["categorical_cols"]) or "_none_")
            g3.markdown("**Date**\n" + "\n".join(f"- {c}" for c in col_types["date_cols"]) or "_none_")
            g4.markdown("**ID**\n" + "\n".join(f"- {c}" for c in col_types["id_cols"]) or "_none_")

        # ---------------- EDA ----------------
        with tab_eda:
            eda_df = df.sample(50_000, random_state=42) if len(df) > 200_000 else df

            analyzable_cols = col_types["numeric_cols"] + col_types["categorical_cols"] + col_types["date_cols"]
            if not analyzable_cols:
                st.info("No numeric, categorical, or date columns available to analyze "
                        "(every column looks like an ID or is constant).")
            else:
                selected_col = st.selectbox("Select column for analysis", analyzable_cols)

                if selected_col in col_types["numeric_cols"]:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(px.histogram(eda_df, x=selected_col, nbins=30,
                                                      title=f"Histogram \u2014 {selected_col}"),
                                         use_container_width=True)
                    with c2:
                        st.plotly_chart(px.box(eda_df, y=selected_col, title=f"Box Plot \u2014 {selected_col}"),
                                         use_container_width=True)
                    st.markdown("**Summary statistics**")
                    st.dataframe(eda_df[selected_col].describe().to_frame(), use_container_width=True)

                elif selected_col in col_types["categorical_cols"]:
                    vc = eda_df[selected_col].value_counts()
                    if len(vc) > 30:
                        st.caption(f"{len(vc)} distinct categories \u2014 showing the top 30.")
                        vc = vc.head(30)
                    vc_df = vc.reset_index()
                    vc_df.columns = [selected_col, "Count"]
                    st.plotly_chart(px.bar(vc_df, x=selected_col, y="Count",
                                            title=f"Category Distribution \u2014 {selected_col}"),
                                     use_container_width=True)
                    st.markdown("**Top categories**")
                    st.dataframe(vc_df, use_container_width=True)

                elif selected_col in col_types["date_cols"]:
                    parsed = pd.to_datetime(eda_df[selected_col], errors="coerce")
                    ts = parsed.dropna()
                    if ts.empty:
                        st.warning("This column could not be parsed as dates for any rows.")
                    else:
                        span_days = (ts.max() - ts.min()).days
                        freq = "W" if span_days <= 120 else "M"
                        counts = ts.dt.to_period(freq).value_counts().sort_index()
                        counts.index = counts.index.astype(str)
                        st.plotly_chart(
                            px.line(x=counts.index, y=counts.values, markers=True,
                                    labels={"x": selected_col, "y": "Record count"},
                                    title=f"Records over time ({'weekly' if freq=='W' else 'monthly'}) \u2014 {selected_col}"),
                            use_container_width=True,
                        )

                st.divider()
                st.subheader("Correlation Analysis")
                if len(col_types["numeric_cols"]) >= 2:
                    corr = ga.correlation_matrix(eda_df, col_types["numeric_cols"])
                    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                               zmin=-1, zmax=1, title="Correlation Heatmap"),
                                     use_container_width=True)
                    strong = ga.strong_correlations(corr, threshold=0.6)
                    if len(strong) > 0:
                        st.markdown("**Strong correlations (|r| \u2265 0.6)**")
                        st.dataframe(strong, use_container_width=True)
                    else:
                        st.caption("No strong correlations (|r| \u2265 0.6) found among numeric columns.")
                    st.caption("Correlation measures association only, not causation.")
                else:
                    st.info("Need at least two numerical columns to compute a correlation matrix.")

        # ---------------- Preprocessing ----------------
        with tab_prep:
            st.subheader("Outlier Detection")
            if col_types["numeric_cols"]:
                method = st.radio("Detection method", ["IQR", "Z-score"], horizontal=True, key="gen_outlier_method")
                outlier_tbl = ga.outlier_summary(df, col_types["numeric_cols"], method=method)
                st.dataframe(outlier_tbl, use_container_width=True)

                cols_with_outliers = outlier_tbl[outlier_tbl["Potential Outliers"] > 0]["Column"].tolist()
                if cols_with_outliers:
                    oc1, oc2 = st.columns(2)
                    outlier_col = oc1.selectbox("Column to act on", cols_with_outliers, key="gen_outlier_col")
                    action = oc2.radio("Action", ["Keep outliers", "Remove outliers", "Cap (winsorize)"],
                                        key="gen_outlier_action")
                    if st.button("Apply outlier action", key="gen_apply_outlier"):
                        if action == "Remove outliers":
                            st.session_state["gen_working_df"] = ga.remove_outliers(df, outlier_col)
                        elif action == "Cap (winsorize)":
                            st.session_state["gen_working_df"] = ga.cap_outliers(df, outlier_col)
                        st.success(f"Applied '{action}' to {outlier_col}.")
                        st.rerun()
                else:
                    st.caption("No potential outliers detected with the selected method.")
            else:
                st.info("No numerical columns available for outlier detection.")

            st.divider()
            st.subheader("Missing Value Handling")
            miss_report = ga.missing_value_report(df, col_types["numeric_cols"], col_types["categorical_cols"])
            if len(miss_report) == 0:
                st.success("No missing values detected \u2014 nothing to handle.")
            else:
                st.dataframe(miss_report, use_container_width=True)
                mc1, mc2 = st.columns(2)
                miss_col = mc1.selectbox("Column to fix", miss_report["Column"].tolist(), key="gen_miss_col")
                is_numeric_col = miss_col in col_types["numeric_cols"]
                strategy_options = (
                    ["mean", "median", "mode", "drop_rows"] if is_numeric_col
                    else ["mode", "unknown", "drop_rows", "ffill"]
                )
                strategy = mc2.selectbox("Strategy", strategy_options, key="gen_miss_strategy")
                if st.button("Apply missing-value strategy", key="gen_apply_missing"):
                    st.session_state["gen_working_df"] = ga.apply_missing_value_strategy(df, miss_col, strategy)
                    st.success(f"Applied '{strategy}' to {miss_col}.")
                    st.rerun()

            st.caption("Changes here update the working dataset used by the Modeling tab below. "
                       "The original uploaded file is never modified.")

        # ---------------- Modeling ----------------
        with tab_model:
            st.subheader("Select Prediction Target")
            candidate_targets = col_types["numeric_cols"] + col_types["categorical_cols"]
            if not candidate_targets:
                st.info("No numeric or categorical columns available to use as a target.")
            else:
                suggested = ga.suggest_target_columns(df, col_types["numeric_cols"])
                if suggested:
                    st.caption(f"Suggested target column(s), based on column name and type: "
                               f"{', '.join(suggested)}. You can select any column below.")

                target_col = st.selectbox(
                    "Target Variable", candidate_targets,
                    index=candidate_targets.index(suggested[0]) if suggested else 0,
                    key="gen_target_select",
                )

                default_problem_type = ga.suggest_problem_type(df, target_col)
                # A `key`ed widget keeps its prior value across reruns even
                # if `index` changes, so re-suggesting on a NEW target
                # requires explicitly resetting the stored value the first
                # time we see that target (the user can still override it
                # afterwards - this only fires once per target change).
                if st.session_state.get("gen_last_target_for_suggestion") != target_col:
                    st.session_state["gen_problem_type_select"] = default_problem_type
                    st.session_state["gen_last_target_for_suggestion"] = target_col
                problem_type = st.radio(
                    "Problem Type", ["Regression", "Classification"],
                    horizontal=True, key="gen_problem_type_select",
                )
                st.caption(f"Suggested based on the target column: **{default_problem_type}** "
                           "\u2014 override above if needed.")

                test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05, key="gen_test_size")

                if st.button("Train Models", type="primary", key="gen_train_btn"):
                    target_missing = df[target_col].isna().sum()
                    if target_missing > 0:
                        st.warning(f"{target_missing} row(s) have a missing target value and will "
                                   "be excluded from training/evaluation.")
                    try:
                        working_df, num_feats, cat_feats = ga.build_feature_lists(df, target_col, col_types)
                        if not num_feats and not cat_feats:
                            st.error("\u26A0\uFE0F No usable feature columns remain after excluding the "
                                      "target and ID/constant columns.")
                        else:
                            results_df, pipelines, X_test, y_test, split = ga.train_and_evaluate(
                                working_df, target_col, num_feats, cat_feats, problem_type, test_size
                            )
                            st.session_state["gen_target"] = target_col
                            st.session_state["gen_problem_type"] = problem_type
                            st.session_state["gen_results_df"] = results_df
                            st.session_state["gen_pipelines"] = pipelines
                            st.session_state["gen_X_test"] = X_test
                            st.session_state["gen_y_test"] = y_test
                            st.session_state["gen_split"] = split
                            best_name = results_df.iloc[0]["Model"]
                            st.session_state["gen_best_model_name"] = best_name
                            st.session_state["gen_importance_df"] = ga.compute_feature_importance(pipelines[best_name])
                            st.success(f"Trained {len(pipelines)} models on "
                                       f"{split[0]:,} training rows / {split[1]:,} test rows.")
                    except ValueError as e:
                        st.error(f"\u26A0\uFE0F {e}")
                    except Exception:
                        st.error("\u26A0\uFE0F Model training failed \u2014 this can happen with very small "
                                  "datasets, a target with too many rare categories, or columns that "
                                  "don't contain usable data. Try a different target or preprocessing step.")

            if st.session_state["gen_results_df"] is not None:
                st.divider()
                st.markdown(f"**Model comparison \u2014 target: `{st.session_state['gen_target']}` "
                           f"({st.session_state['gen_problem_type']})**")
                st.dataframe(st.session_state["gen_results_df"], use_container_width=True)
                st.caption(f"Best model: **{st.session_state['gen_best_model_name']}**")

                if st.session_state["gen_problem_type"] == "Classification":
                    best_pipe = st.session_state["gen_pipelines"][st.session_state["gen_best_model_name"]]
                    preds = best_pipe.predict(st.session_state["gen_X_test"])
                    cm = ga.confusion_matrix_df(st.session_state["gen_y_test"], preds)
                    st.markdown("**Confusion Matrix (best model)**")
                    st.dataframe(cm, use_container_width=True)

        # ---------------- Insights ----------------
        with tab_insights:
            if st.session_state["gen_results_df"] is None:
                st.info("Train a model in the Modeling tab to see feature importance, predictions, "
                        "and downloadable results here.")
            else:
                imp_df = st.session_state["gen_importance_df"]
                if imp_df is not None and len(imp_df) > 0:
                    st.subheader("Feature Importance")
                    st.plotly_chart(
                        px.bar(imp_df.head(15).sort_values("Importance"), x="Importance", y="Feature",
                               orientation="h", title=f"Top Features \u2014 {st.session_state['gen_best_model_name']}"),
                        use_container_width=True,
                    )
                    st.caption("These features were most influential in the model's predictions. "
                               "This reflects association within the model, not proven causation.")
                else:
                    st.info(f"{st.session_state['gen_best_model_name']} does not expose feature "
                            "importances (e.g. linear/logistic models) \u2014 try Random Forest or "
                            "Gradient Boosting for an importance ranking.")

                st.divider()
                st.subheader("Download Results")
                d1, d2, d3 = st.columns(3)
                d1.download_button(
                    "\u2B07\uFE0F Cleaned dataset (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="cleaned_dataset.csv", mime="text/csv",
                )
                d2.download_button(
                    "\u2B07\uFE0F Data quality summary (CSV)",
                    ga.data_quality_table(df).to_csv(index=False).encode("utf-8"),
                    file_name="data_quality_summary.csv", mime="text/csv",
                )
                d3.download_button(
                    "\u2B07\uFE0F Model metrics (CSV)",
                    st.session_state["gen_results_df"].to_csv(index=False).encode("utf-8"),
                    file_name="model_metrics.csv", mime="text/csv",
                )

                d4, d5 = st.columns(2)
                best_pipe = st.session_state["gen_pipelines"][st.session_state["gen_best_model_name"]]
                preds_out = st.session_state["gen_X_test"].copy()
                preds_out["Actual"] = st.session_state["gen_y_test"].values
                preds_out["Predicted"] = best_pipe.predict(st.session_state["gen_X_test"])
                d4.download_button(
                    "\u2B07\uFE0F Predictions (CSV)",
                    preds_out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv", mime="text/csv",
                )
                if imp_df is not None and len(imp_df) > 0:
                    d5.download_button(
                        "\u2B07\uFE0F Feature importance (CSV)",
                        imp_df.to_csv(index=False).encode("utf-8"),
                        file_name="feature_importance.csv", mime="text/csv",
                    )

    st.divider()
    st.caption("Privacy note: uploaded files are processed only in this session/container, are never "
               "sent to an external API, and are not permanently stored beyond this app session.")
