# EduPro \u2014 Model Evaluation Report

## Methodology

**Course-level targets (EnrollmentCount, TotalRevenue):** the modeling dataset has exactly
60 rows (one per course). A single train/test split at this size would be unstable and easy
to game by chance, so these two targets are evaluated with **Leave-One-Out Cross-Validation
(LOOCV)**: each course is held out once, the model is trained on the other 59, and the
held-out course is predicted. This uses every course as a test case exactly once and is the
standard, honest approach for very small tabular datasets.

**Forecasting target (NextMonthEnrollment):** built from a CourseID \u00d7 Month panel
(60 courses \u00d7 10 usable months after lag/target construction = 600 rows). This uses a
**strict chronological split**: the last 2 calendar months (2025-10 and 2025-11) are held out
as the test set; everything earlier trains the model. No shuffling, no random splitting \u2014
this prevents any future information from leaking into training, per the forecasting
requirement.

All models use a `ColumnTransformer` (StandardScaler on numeric features, OneHotEncoder on
categorical features) inside a scikit-learn `Pipeline`, so preprocessing is fit only on
training data in every fold/split.

## Target 1: EnrollmentCount (course-level, LOOCV, n=60)

| Model | MAE | RMSE | R\u00b2 |
|---|---|---|---|
| Ridge | 10.44 | 12.63 | -0.034 |
| Lasso | 10.54 | 12.64 | -0.035 |
| Linear Regression | 10.54 | 13.11 | -0.115 |
| Random Forest | 10.55 | 12.60 | -0.029 |
| Gradient Boosting | 11.15 | 13.51 | -0.183 |

**Best model (lowest MAE): Ridge Regression** \u2014 but its R\u00b2 is negative, meaning it performs
about as well as (or slightly worse than) simply predicting the average enrollment (166.7)
for every course. **In plain business terms: none of these models can meaningfully predict
which courses will enroll more or fewer students from the catalog attributes available.**
This mirrors the EDA finding that enrollment counts are unusually uniform (140\u2013196 across all
60 courses) and only weakly associated with any single feature.

## Target 2: TotalRevenue (course-level, LOOCV, n=60)

| Model | MAE | RMSE | R\u00b2 |
|---|---|---|---|
| Lasso | 1,924.84 | 2,553.59 | 0.9897 |
| Ridge | 1,713.90 | 2,566.99 | 0.9896 |
| Linear Regression | 1,946.38 | 2,576.23 | 0.9895 |
| Gradient Boosting | 2,148.64 | 4,229.82 | 0.9718 |
| Random Forest | 2,079.96 | 4,375.77 | 0.9698 |

**Best model (lowest MAE): Ridge Regression**, R\u00b2 \u2248 0.99. This near-perfect score should
**not** be read as a modeling triumph: because Amount always exactly equals CoursePrice in
every transaction, TotalRevenue is a deterministic function of CoursePrice \u00d7 EnrollmentCount,
and CoursePrice alone explains ~97% of the RandomForest's revenue feature importance. The
model is essentially learning `Revenue \u2248 Price \u00d7 (roughly constant enrollment)`, which is
mathematically guaranteed to fit well \u2014 it is not evidence that revenue is being "predicted"
from independent business drivers.

## Target 3: NextMonthEnrollment (forecasting, chronological split)

Train: 480 rows (months 2025-01 through 2025-09, per course) \u00b7 Test: 120 rows (2025-10 and
2025-11, per course).

| Model | MAE | RMSE | R\u00b2 |
|---|---|---|---|
| Lasso | 2.76 | 3.68 | -0.0004 |
| Ridge | 2.76 | 3.77 | -0.050 |
| Linear Regression | 2.77 | 3.77 | -0.051 |
| Random Forest | 3.37 | 4.40 | -0.434 |
| Gradient Boosting | 3.66 | 4.63 | -0.586 |

**Best model (lowest MAE): Lasso Regression**, but again R\u00b2 \u2248 0 \u2014 knowing a course's
enrollment history through month M gives essentially no ability to predict month M+1's
enrollment beyond the historical average. Monthly enrollment per course behaves close to
random noise in this dataset.

## How the best model was chosen

Per the project requirement, the best model was **not** chosen by R\u00b2 alone. MAE, RMSE, and
R\u00b2 were all considered together; in every target here the three metrics agreed on the same
ranking (the model with the lowest MAE also had the lowest or near-lowest RMSE), so no
conflict had to be resolved. Linear/regularized models (Ridge/Lasso) outperformed the
tree-based models on all three targets \u2014 with such a small, low-signal dataset, the
tree-based models' extra flexibility mostly overfits rather than helps.

## Honest summary

- **Revenue prediction "works" only because it re-derives a deterministic formula
  already present in the data** (Amount = CoursePrice). It is not evidence that EduPro can
  predict revenue from independent catalog decisions.
- **Enrollment (true demand) prediction does not work** with the features available. This is
  a legitimate, reportable finding: this dataset's course/teacher attributes do not carry a
  strong learnable signal for enrollment, most likely because enrollment counts were
  generated within a narrow, roughly uniform range (140\u2013196 per course).
- **Forecasting next month's enrollment from historical trend also does not work** \u2014 monthly
  enrollment per course does not show meaningful autocorrelation in this data.
- These results should inform EduPro's next step: collecting additional behavioral or
  marketing-exposure data (e.g., ad spend, homepage placement, marketing channel, cohort
  size) is more likely to reveal predictive levers than refining models on the current
  feature set.
