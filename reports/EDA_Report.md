# EduPro \u2014 Exploratory Data Analysis Report

Dataset: `EduPro_Online_Platform.xlsx` (Users: 3,000 rows \u00b7 Teachers: 60 rows \u00b7 Courses: 60 rows \u00b7 Transactions: 10,000 rows)
Charts referenced below are saved in `reports/figures/`.

## A\u2013C. Dataset overview, missing values, duplicates

No missing values and no duplicate rows exist in any of the four sheets. Every ID column (UserID, TeacherID, CourseID, TransactionID) is 100% unique within its sheet, and every foreign key in Transactions resolves to a valid parent row. This is a clean synthetic dataset with no data-entry noise to repair.

## D\u2013F. Categorical distributions (course category, type, level)

- **Category** (`D_category_distribution.png`): 12 categories, each represented by roughly 4\u20136 courses out of 60 \u2014 the catalog is fairly evenly spread across subject areas, no single category dominates the course count.
- **Type** (`E_coursetype_distribution.png`): 38 of 60 courses (63%) are Free, 22 (37%) are Paid.
- **Level** (`F_courselevel_distribution.png`): Beginner/Intermediate/Advanced are represented, without one level dominating the catalog.

**Business read:** EduPro's catalog is broad rather than concentrated \u2014 no category or level is being neglected in terms of course *count*, though (as shown below) that doesn't mean every category performs equally in enrollment or revenue.

## G\u2013K. Numeric distributions (price, duration, rating, teacher experience/rating)

- **Price** (`G_price_distribution.png`): heavily right-skewed with a spike at $0 (the 38 free courses); paid courses range up to $490.90, mean price across all courses \u2248 $93.
- **Duration** (`H_duration_distribution.png`): ranges 1.2\u201349.7 hours, fairly spread out with no single typical length.
- **Course rating** (`I_courserating_distribution.png`): spans 1.13\u20134.94, roughly centered in the 3\u20134 range.
- **Teacher experience per course** (`J_teacher_experience_distribution.png`, averaged across each course's instructor pool): mostly clustered 10\u201320 years.
- **Teacher rating per course** (`K_teacher_rating_distribution.png`): mostly clustered around 3.5\u20134.5.

**Business read:** the catalog covers a wide price and duration range, and the instructor pool skews toward experienced, well-rated teachers on average.

## L\u2013M. Enrollment and revenue distributions

- **Enrollment** (`L_enrollment_distribution.png`): tight range of 140\u2013196 enrollments per course (mean 166.7, std 12.5) \u2014 enrollment is unusually *uniform* across the catalog; no course is a dramatic outlier in raw popularity.
- **Revenue** (`M_revenue_distribution.png`): far more skewed (mean \\$15,189, but 25th percentile is \\$0 because every free course contributes \\$0) \u2014 revenue variation is driven almost entirely by price, not by enrollment variation.

**Business read:** because enrollment counts are so evenly spread, revenue differences between courses come from **pricing**, not from some courses being dramatically more "in demand" than others in this dataset.

## N\u2013O. Enrollment and revenue by category

- **Enrollment by category** (`N_enrollment_by_category.png`): Data Science leads with 916 total enrollments, followed by Finance (864) and Web Development (844).
- **Revenue by category** (`O_revenue_by_category.png`): Artificial Intelligence leads with \\$202,750.67, followed by Business (\\$181,527.58) and Project Management (\\$169,103.21).

**Business read:** the top category by enrollment (Data Science) is *not* the top category by revenue (Artificial Intelligence) \u2014 a category's popularity and its revenue contribution are driven by different factors (how many people enroll vs. what price its courses carry).

## P\u2013Q. Top 10 courses

`P_top10_enrollment.png` and `Q_top10_revenue.png` show the highest-performing individual courses on each metric respectively \u2014 the two lists only partially overlap, again illustrating that price, not just popularity, determines revenue leadership.

## R\u2013V. Relationship charts

- **Price vs. enrollment** (`R_price_vs_enrollment.png`): weak negative correlation (r \u2248 \u22120.16). Higher-priced courses do not show meaningfully lower (or higher) enrollment in this data.
- **Rating vs. enrollment** (`S_rating_vs_enrollment.png`): weak positive correlation (r \u2248 0.29) \u2014 the strongest single numeric association with enrollment found in this dataset, but still modest.
- **Duration vs. enrollment** (`T_duration_vs_enrollment.png`): essentially no visible relationship (r \u2248 \u22120.10).
- **Teacher rating vs. enrollment** (`U_teacherrating_vs_enrollment.png`): negligible relationship (r \u2248 0.02).
- **Teacher experience vs. enrollment** (`V_teacherexperience_vs_enrollment.png`): negligible relationship (r \u2248 0.07).

**Business read:** none of the catalog attributes examined show a strong association with enrollment. Course rating shows the clearest (if still weak) positive signal. This is an important, honest finding carried through to the modeling phase: **enrollment in this dataset does not appear strongly driven by price, duration, or instructor characteristics.**

## W\u2013X. Monthly trends

- **Monthly enrollment** (`W_monthly_enrollment_trend.png`): ranges from 762 to 899 transactions/month across the full 2025 calendar year \u2014 mild month-to-month variation, no dramatic seasonal spikes or collapses.
- **Monthly revenue** (`X_monthly_revenue_trend.png`): tracks enrollment volume closely, as expected given revenue = price \u00d7 enrollment.

## Y. Category-wise revenue trends

`Y_category_revenue_trends.png` shows the top 5 revenue categories' monthly revenue over the year \u2014 each category's monthly revenue fluctuates with its own transaction volume, without a consistent category pulling away over time.

## Z. Correlation analysis

`Z_correlation_heatmap.png` summarizes correlations across CoursePrice, CourseDuration, CourseRating, EnrollmentCount, TotalRevenue, AvgTeacherRating, AvgTeacherExperience, DistinctTeacherCount, and RecencyDays. The standout result: **CoursePrice and TotalRevenue correlate at r = 0.997** \u2014 this is a near-exact mechanical relationship, not a discovered business insight, because Amount always exactly equals CoursePrice in every transaction (see Phase 1 data-quality note). By contrast, every correlation involving EnrollmentCount is weak (|r| < 0.3), reinforcing that demand is the harder, less-explained target in this dataset.

## Correlation is not causation

All relationships above describe statistical association only. None of them establish that, for example, raising a course's rating *causes* higher enrollment \u2014 only that the two vary together weakly in the historical data.
