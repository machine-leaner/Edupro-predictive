# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

## 1. Abstract

Online learning platforms increasingly need to move from descriptive reporting to
predictive analytics to guide catalog and pricing decisions. This paper presents an
end-to-end machine learning study of EduPro, an online course platform, using its actual
operational data (3,000 users, 60 teachers, 60 courses, 10,000 transactions across the 2025
calendar year). We build a course-level analytical dataset, conduct exploratory analysis,
engineer catalog and instructor-pool features, and train five regression algorithms
(Linear, Ridge, Lasso, Random Forest, Gradient Boosting) to predict course enrollment
(demand) and revenue. We find that revenue is near-perfectly predictable (R\u00b2 \u2248 0.99) only
because it is a deterministic function of price in this dataset, while enrollment \u2014 the
genuine demand signal EduPro cares about \u2014 shows essentially no predictive skill above the
historical mean (R\u00b2 \u2248 0) across every model tested, including a chronologically-validated
next-month forecasting variant. We report this honestly as a substantive finding about the
current dataset's limits rather than presenting an inflated headline result, and translate
it into concrete recommendations for what EduPro should collect next.

## 2. Introduction

EduPro operates an online course marketplace spanning 12 subject categories. Historically,
decisions about which courses to promote, price, or expand have relied on retrospective
reporting. This project investigates whether EduPro's existing data \u2014 course attributes,
instructor characteristics, and transaction history \u2014 can support predictive models of
future enrollment and revenue, and quantifies how much signal is actually present.

## 3. Problem Statement

Given a course's catalog attributes (category, type, level, price, duration, rating) and
its associated instructor pool's characteristics (experience, rating), can EduPro reliably
predict (a) how many students will enroll and (b) how much revenue the course will generate,
using only historically observed data and without assuming relationships the data does not
support?

## 4. Objectives

1. Build a clean, leakage-aware, course-level modeling dataset from four raw source tables.
2. Characterize demand and revenue patterns via exploratory data analysis.
3. Engineer features that are genuinely available before/at the point of prediction.
4. Train and fairly compare interpretable and nonlinear regression models.
5. Quantify feature importance and translate results into business recommendations,
   including an honest account of where the data does not support strong conclusions.

## 5. Dataset Description

The raw workbook contains four sheets. **Users** (3,000 rows: UserID, UserName, Age,
Gender, Email) is demographic reference data. **Teachers** (60 rows: TeacherID, Age,
Gender, Expertise, YearsOfExperience, TeacherRating). **Courses** (60 rows: CourseID,
CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating).
**Transactions** (10,000 rows: TransactionID, UserID, CourseID, TeacherID,
TransactionDate, Amount, PaymentMethod), spanning 2025-01-01 to 2025-12-30.

No missing values or duplicate rows were found in any sheet, and all foreign keys in
Transactions resolve to valid parent records. Two structural properties of the data were
identified during inspection and shape every subsequent phase: (i) Amount is always exactly
equal to the referenced course's CoursePrice (0 exceptions across 10,000 transactions), and
(ii) the CourseID\u2013TeacherID relationship is many-to-many and exists only within
Transactions \u2014 each course is taught, across its history, by between 7 and 30 distinct
teachers.

## 6. Data Preprocessing

Dates were parsed to datetime64 (100% success). Categorical columns (CourseCategory,
CourseType, CourseLevel, Expertise, Gender, PaymentMethod) were cast to categorical dtype.
Numeric ranges were validated (no negative prices, durations, or ratings; ratings fall
within the documented 1\u20135 scale). CourseType ("Free"/"Paid") was cross-checked against
CoursePrice for consistency \u2014 no mismatches were found. Free-course enrollments (Amount = 0)
were flagged via an `IsPaidTransaction` indicator rather than removed, since they are real
enrollment events relevant to demand modeling, just not to paid-revenue analysis.

## 7. Exploratory Data Analysis

Enrollment per course is unusually uniform: 140\u2013196 transactions per course (mean 166.7,
std 12.5) across all 60 courses. Revenue per course is far more dispersed (driven by price)
and correlates with CoursePrice at r = 0.997 \u2014 a near-exact mechanical relationship rather
than a discovered pattern. Among catalog attributes, only CourseRating shows a non-trivial
association with enrollment (r = 0.294); CoursePrice (r = \u22120.163), CourseDuration
(r = \u22120.103), average instructor rating (r = 0.019), and average instructor experience
(r = 0.073) all show weak-to-negligible correlation with enrollment. By category, Data
Science leads total enrollment (916), while Artificial Intelligence leads total revenue
($202,750.67) \u2014 the divergence between the two leaderboards illustrates that revenue
leadership in this dataset reflects pricing rather than popularity. Monthly transaction
volume across 2025 ranges narrowly from 762 to 899, showing mild but no dramatic
seasonality. Full chart set and category-by-category detail are in `EDA_Report.md`.

## 8. Feature Engineering

Because Courses and Teachers have no static link, teacher-related features were computed
as historical aggregates over each course's observed instructor pool (mean/max rating,
mean experience, distinct-teacher count) rather than a one-to-one join. Course-level
transaction aggregates (EnrollmentCount, TotalRevenue, AverageTransactionAmount,
RevenuePerEnrollment) were computed with GroupBy on CourseID. An enrollment/revenue trend
feature compared each course's own first-half vs. second-half transaction history (using
only that course's own past data). Category-level historical demand/revenue features used
leave-one-out category means so a course's own outcome could never leak into its own
feature. A separate CourseID\u00d7Month panel (Target 3) was built with lag-1 and rolling-3-month
features computed strictly from prior months, to support a genuinely forward-looking
forecasting test.

## 9. Methodology

Two evaluation regimes were used, chosen for defensibility rather than convenience.
Course-level targets (EnrollmentCount, TotalRevenue) have only 60 observations \u2014 too few
for a stable single train/test split \u2014 so Leave-One-Out Cross-Validation was used, giving
every course a turn as the held-out test case. The forecasting target
(NextMonthEnrollment) was evaluated with a strict chronological split: months 1\u20139 (480
course-month rows) trained the models, months 10\u201311 (120 rows) tested them, with no
shuffling, preventing any future information from entering training.

## 10. Machine Learning Algorithms

Three interpretable baselines (Linear Regression, Ridge, Lasso) and two nonlinear ensemble
models (Random Forest Regressor, Gradient Boosting Regressor) were trained per target, each
inside a scikit-learn `Pipeline` with a `ColumnTransformer` (StandardScaler for numeric
features, OneHotEncoder for categorical features) to prevent preprocessing leakage across
folds/splits.

## 11. Experimental Setup

All models used identical feature sets per target for a fair comparison: CoursePrice,
CourseDuration, CourseRating, AvgTeacherRating, AvgTeacherExperience, DistinctTeacherCount
(numeric) plus CourseCategory, CourseType, CourseLevel (categorical) for the course-level
targets; Lag1Enrollment, Lag1Revenue, RollingAvgEnrollment3M, CumulativeEnrollmentToDate,
CoursePrice, CourseDuration, CourseRating (numeric) plus the same three categoricals for the
forecasting target. Random seeds were fixed (random_state=42) for reproducibility.

## 12. Evaluation Metrics

Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R\u00b2 were computed for every
model/target combination. The best model per target was selected by lowest MAE, with RMSE
consulted as a tiebreaker, rather than by R\u00b2 alone, per the project's evaluation
requirement.

## 13. Results

**EnrollmentCount (LOOCV, n=60):** best model Ridge (MAE 10.44, RMSE 12.63, R\u00b2 \u22120.034);
all five models produced R\u00b2 at or below zero. **TotalRevenue (LOOCV, n=60):** best model
Ridge (MAE 1,713.90, RMSE 2,566.99, R\u00b2 0.990); all models scored R\u00b2 \u2265 0.97.
**NextMonthEnrollment (chronological split):** best model Lasso (MAE 2.76, RMSE 3.68,
R\u00b2 \u22120.0004); all models scored R\u00b2 at or below zero. Full per-model tables are in
`Model_Evaluation.md`.

## 14. Feature Importance

RandomForest impurity-based importance for EnrollmentCount ranks CourseRating (0.233) and
CourseDuration (0.208) highest, followed by DistinctTeacherCount (0.109) and
AvgTeacherExperience (0.095). For TotalRevenue, CoursePrice dominates at 0.971 importance,
with every other feature below 0.01 \u2014 a direct consequence of Amount always equalling
CoursePrice, not an independently discovered pricing insight. Importance reflects
association within the fitted model, not a causal claim.

## 15. Business Insights

Revenue leadership (Artificial Intelligence) and enrollment leadership (Data Science) are
different categories, showing that price and popularity are separate levers at EduPro.
Course rating is the only catalog attribute with a non-trivial (if still weak) positive
association with enrollment; price, duration, and instructor characteristics show little to
no association with enrollment in this dataset. Because revenue is mechanically
price \u00d7 enrollment here, EduPro can compute exact revenue scenarios for a given price once
an enrollment estimate is available \u2014 but the data does not support a strong claim that
discounting would substantially grow enrollment (price/enrollment r \u2248 \u22120.16). Monthly
enrollment shows only mild seasonality across the observed year.

## 16. Limitations

The dataset provides only 60 courses and 60 teachers, which is too small for a stable
holdout split and limits statistical power throughout; LOOCV mitigates but does not
eliminate this constraint. TotalRevenue carries no information beyond price \u00d7 enrollment
because Amount always equals CoursePrice with no discount/refund variation recorded, so
"revenue prediction" in this dataset cannot be validated as an independent business skill.
Enrollment counts are unusually uniform across courses (140\u2013196), which caps how much
variance any model can explain from catalog attributes. Teacher effects are only
observable as a pooled, historical instructor-pool average per course, not a true
per-instructor effect, because Courses and Teachers have no static link.

## 17. Future Scope

Future work should prioritize collecting features more likely to explain enrollment
variance \u2014 marketing spend, traffic source, homepage/search placement, promotional
campaigns, and cohort-level engagement signals \u2014 since the current catalog/instructor
attributes carry little demand signal. Recording actual transaction discounts separately
from list price would let revenue modeling reflect genuine price elasticity rather than a
fixed formula. Expanding the course catalog would allow a conventional train/test split and
more statistically powerful models. If EduPro moves to a fixed primary-instructor-per-course
structure, a true (non-pooled) teacher effect could then be tested.

## 18. Conclusion

This project delivers a complete, leakage-aware pipeline from raw EduPro data to a working
predictive dashboard, and — just as importantly — an honest empirical account of what the
current data can and cannot support. Revenue is trivially well-fit due to a deterministic
relationship in the data, not genuine predictive power; enrollment/demand, the metric EduPro
actually needs to forecast, is not well-explained by the currently available catalog and
instructor attributes. This is a legitimate and actionable finding: it tells EduPro where to
invest in additional data collection rather than in further model tuning on the current
feature set.
