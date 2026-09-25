# EduPro \u2014 Predictive Modeling for Course Demand and Revenue Forecasting

An end-to-end data science project that analyzes EduPro's real course, teacher, user, and
transaction data and builds machine learning models to predict course enrollment demand and
revenue, delivered through an interactive Streamlit dashboard.

## Overview

EduPro is an online learning platform moving from historical reporting to predictive
analytics. This project ingests EduPro's four core data tables (Users, Teachers, Courses,
Transactions), builds a clean course-level analytical dataset, explores it thoroughly, trains
and compares five regression algorithms per target, and exposes the results through a
9-page Streamlit application.

## Problem Statement

EduPro wants to know, ahead of time: which courses/categories are likely to attract strong
enrollment, which are likely to generate strong revenue, and what catalog or instructor
attributes (price, duration, rating, teacher experience/rating) are associated with those
outcomes \u2014 using only its own historical data, without inventing patterns that aren't there.

## Objectives

1. Predict course enrollment (demand).
2. Predict course revenue.
3. Surface category-level demand/revenue trends.
4. Quantify how price, rating, duration, and instructor characteristics relate to demand.
5. Deliver actionable, data-grounded business recommendations via a dashboard.

## Dataset

`data/EduPro_Online_Platform.xlsx`, 4 sheets:

| Sheet | Rows | Key columns |
|---|---|---|
| Users | 3,000 | UserID, Age, Gender |
| Teachers | 60 | TeacherID, Expertise, YearsOfExperience, TeacherRating |
| Courses | 60 | CourseID, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating |
| Transactions | 10,000 | TransactionID, UserID, CourseID, TeacherID, TransactionDate, Amount, PaymentMethod |

No missing values or duplicates in any sheet; all foreign keys resolve cleanly. Two
dataset characteristics shape the whole project and are documented throughout:

- **Amount always exactly equals the course's CoursePrice** \u2014 there is no discount/refund
  variance, so `TotalRevenue = CoursePrice \u00d7 EnrollmentCount` deterministically.
- **Courses and Teachers have no fixed 1:1 relationship** \u2014 it only exists inside
  Transactions, and it's many-to-many (every course is taught by 7\u201330 different instructors
  across its history). Teacher features are therefore computed as historical aggregates over
  each course's instructor pool, not a static join.

## Architecture

```
edupro-predictive-modeling/
\u251c\u2500\u2500 data/
\u2502   \u251c\u2500\u2500 EduPro_Online_Platform.xlsx        # raw, untouched
\u2502   \u2514\u2500\u2500 processed/                          # cleaned CSVs + modeling datasets
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 data_loader.py                      # Phase 1: load + quality summary
\u2502   \u251c\u2500\u2500 preprocessing.py                    # Phase 2: cleaning, validation
\u2502   \u251c\u2500\u2500 feature_engineering.py              # Phase 3/4/6: aggregation, merge, features
\u2502   \u251c\u2500\u2500 eda.py                              # Phase 5: charts + findings
\u2502   \u251c\u2500\u2500 train_models.py                     # Phase 8/9: pipelines, training
\u2502   \u251c\u2500\u2500 evaluate_models.py                  # Phase 11: feature importance
\u2502   \u2514\u2500\u2500 predictions.py                      # model-loading helper for the app
\u251c\u2500\u2500 models/                                 # saved model pipelines (.pkl) + metadata
\u251c\u2500\u2500 reports/
\u2502   \u251c\u2500\u2500 figures/                            # all EDA + feature-importance charts
\u2502   \u251c\u2500\u2500 EDA_Report.md
\u2502   \u251c\u2500\u2500 Model_Evaluation.md
\u2502   \u2514\u2500\u2500 *_comparison.csv, *_feature_importance.csv
\u251c\u2500\u2500 app.py                                  # Streamlit dashboard (9 pages)
\u251c\u2500\u2500 requirements.txt
\u2514\u2500\u2500 README.md
```

## EDA Highlights

- Enrollment per course is unusually uniform (140\u2013196, mean 166.7) \u2014 no dramatic outliers.
- Revenue per course is highly skewed and driven almost entirely by price (r = 0.997 with
  CoursePrice), not by enrollment variance (r = \u22120.12 with EnrollmentCount).
- The only numeric feature with a non-trivial (still weak) association with enrollment is
  CourseRating (r = 0.29). Price, duration, and teacher rating/experience show negligible
  correlation with enrollment.
- Data Science leads in total enrollment (916); Artificial Intelligence leads in total
  revenue ($202,750.67) \u2014 the two leaderboards diverge because revenue leadership is a price
  effect, not a popularity effect.

Full write-up: `reports/EDA_Report.md`. All charts: `reports/figures/`.

## Feature Engineering

- **Course-level features:** CoursePrice, CourseDuration, CourseRating, CourseCategory,
  CourseType, CourseLevel, plus engineered PriceBand/DurationBucket/RatingTier.
- **Teacher-pool features (per course, historical aggregate):** AvgTeacherRating,
  AvgTeacherExperience, DistinctTeacherCount, MaxTeacherRating \u2014 computed because no static
  course\u2192teacher join exists.
- **Transaction aggregates:** EnrollmentCount, TotalRevenue, AverageTransactionAmount,
  RevenuePerEnrollment, EnrollmentTrend, RevenueTrend (first-half vs second-half of each
  course's own history \u2014 no future/other-course information used).
- **Category-level historical demand/revenue:** leave-one-out category means, so a course's
  own outcome never leaks into its own feature.
- **Forecasting panel (Target 3):** CourseID \u00d7 Month grid with lag-1 and rolling-3-month
  features computed strictly from prior months, target = next month's enrollment/revenue.

## ML Algorithms

Baselines: Linear Regression, Ridge, Lasso. Advanced: Random Forest Regressor, Gradient
Boosting Regressor. All wrapped in a scikit-learn `ColumnTransformer` + `Pipeline`
(StandardScaler + OneHotEncoder).

## Evaluation Methodology

- **Course-level targets (n=60):** Leave-One-Out Cross-Validation (a single holdout split
  would be unstable at this sample size).
- **Forecasting target:** chronological split (train on months 1\u20139, test on months 10\u201311) \u2014
  no shuffling, no leakage from the future.
- Metrics: MAE, RMSE, R\u00b2, compared side-by-side; the "best" model is chosen by MAE with RMSE
  as a tiebreaker, not by R\u00b2 alone.

## Results (see `reports/Model_Evaluation.md` for full detail and honest caveats)

| Target | Best model | MAE | RMSE | R\u00b2 |
|---|---|---|---|---|
| EnrollmentCount (demand) | Ridge | 10.44 | 12.63 | -0.03 |
| TotalRevenue | Ridge | 1,713.90 | 2,566.99 | 0.99 |
| NextMonthEnrollment (forecast) | Lasso | 2.76 | 3.68 | -0.0004 |

**Read this honestly:** the revenue R\u00b2 is high because revenue is mechanically determined by
price in this dataset (Amount always equals CoursePrice) \u2014 it is not evidence of strong,
independently-learned revenue drivers. The enrollment/demand models, by contrast, show
essentially no predictive skill above the historical average \u2014 a genuine finding about the
limits of this dataset, not a bug.

## Feature Importance

RandomForest impurity-based importance (association, not causation):
- **Enrollment:** CourseRating (0.23) and CourseDuration (0.21) rank highest.
- **Revenue:** CoursePrice (0.97) dominates, a mechanical consequence of Amount = CoursePrice.

Charts: `reports/figures/enrollment_feature_importance.png`,
`reports/figures/revenue_feature_importance.png`.

## Streamlit Application

10 pages: Executive Dashboard \u00b7 Dataset Overview \u00b7 Course Demand Analysis \u00b7 Revenue Analysis \u00b7
Demand Prediction \u00b7 Revenue Prediction \u00b7 Category Analysis \u00b7 Feature Importance \u00b7 Business
Recommendations \u00b7 **Upload & Analyze Dataset**. The EduPro-specific pages load pre-trained
models from `models/` \u2014 they never retrain on page load.

### Mode 2: Upload & Analyze Dataset (generic analyzer)

A second, fully independent mode lets you upload **any** CSV/XLS/XLSX file and get automatic
profiling, EDA, preprocessing, ML modeling, and downloadable results \u2014 without touching the
EduPro-specific code or data. Logic lives in `src/generic_analyzer.py`.

Workflow (`Overview \u2192 Data Quality \u2192 EDA \u2192 Preprocessing \u2192 Modeling \u2192 Insights` tabs):

1. **Upload** a `.csv`, `.xlsx`, or `.xls` file (Excel sheet picker appears automatically for
   multi-sheet workbooks).
2. **Overview / Data Quality**: row/column counts, dtypes, memory usage, missing values,
   duplicates, and automatic column-type detection (numeric / categorical / date / ID /
   constant \u2014 ID and constant columns are excluded from modeling automatically).
3. **EDA**: pick any column for a histogram+box plot (numeric), value-count bar chart
   (categorical), or time trend (date); a correlation heatmap and strong-correlation table
   appear automatically when \u2265 2 numeric columns exist.
4. **Preprocessing**: outlier detection (IQR or Z-score) with keep/remove/cap actions, and a
   missing-value report with mean/median/mode/drop/forward-fill/"Unknown" strategies \u2014
   nothing is changed without an explicit button click.
5. **Modeling**: select a target (name-based suggestions shown, e.g. columns containing
   "revenue", "amount", "score"), confirm/override the suggested problem type (regression vs.
   classification), and train Linear/Ridge + Random Forest + Gradient Boosting (or Logistic
   Regression + Random Forest + Gradient Boosting for classification) with one click. Metrics:
   MAE/RMSE/R\u00b2 for regression, Accuracy/Precision/Recall/F1 + confusion matrix for
   classification.
6. **Insights**: feature importance chart for the best tree-based model, plus download buttons
   for the cleaned dataset, data-quality summary, model metrics, predictions, and feature
   importance \u2014 all as CSV.

Uploaded files are processed only in-session (never sent to an external API and not
persisted beyond the session), and the whole workflow uses `st.session_state` so your
uploaded data, cleaning choices, and trained models survive switching between the tabs
above.

### Installation

```bash
cd edupro-predictive-modeling
pip install -r requirements.txt
```

### How to run

```bash
# 1. Re-run the pipeline (optional - pre-computed artifacts are already included)
cd src
python data_loader.py
python preprocessing.py
python feature_engineering.py
python eda.py
python train_models.py
python evaluate_models.py

# 2. Launch the dashboard
cd ..
streamlit run app.py
```

## Limitations

- Only 60 courses and 60 teachers \u2014 too small a sample for a stable train/test split on
  course-level targets; LOOCV was used instead, but conclusions should be treated as
  preliminary until more courses are added.
- Revenue has no independent signal beyond price \u00d7 enrollment in this dataset (no
  discounts, promotions, or refunds are represented).
- Enrollment counts are unusually uniform across courses, which limits how much any model
  can learn about genuine demand drivers from catalog attributes alone.
- The course\u2194teacher relationship is many-to-many and only observable through
  Transactions, so "teacher effect" here reflects an instructor-pool average, not a single
  instructor's true effect.

## Future Enhancements

- Collect richer behavioral/marketing features (traffic source, homepage placement, ad
  spend, cohort/marketing campaign) that are more likely to explain enrollment variance.
- Track actual discounts/promotions separately from list price so revenue modeling reflects
  real elasticity rather than a fixed formula.
- Expand the catalog size to make a genuine train/test split (rather than LOOCV) feasible.
- Track a fixed primary-instructor assignment per course, if that becomes true of the
  business, to test genuine (not pooled) teacher effects.
