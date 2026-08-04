"""
Lesson 07 : Automated EDA & Business Storytelling

Topics Covered
---------------
1. Automated EDA
2. ydata-profiling
3. Sweetviz
4. Business Storytelling
5. Executive Summary

Dataset:
--------
Titanic Dataset

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import pandas as pd
import seaborn as sns

from ydata_profiling import ProfileReport
import sweetviz as sv

# ==========================================================
# CREATE REPORT DIRECTORY
# ==========================================================

os.makedirs("reports", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 80)
print("AUTOMATED EXPLORATORY DATA ANALYSIS")
print("=" * 80)

df = sns.load_dataset("titanic")

print("\nDataset Loaded Successfully.")

print("\nShape :", df.shape)

# ==========================================================
# BASIC OVERVIEW
# ==========================================================

print("\n========== BASIC OVERVIEW ==========")

print(df.head())

print("\nMissing Values")

print(df.isnull().sum())

print("\nDuplicate Rows")

print(df.duplicated().sum())

# ==========================================================
# YDATA PROFILING
# ==========================================================

print("\n========== YDATA PROFILING ==========")

print("Generating HTML report...")

profile = ProfileReport(

    df,

    title="Titanic Dataset Profiling Report",

    explorative=True

)

profile.to_file("reports/titanic_profile.html")

print("Report Saved Successfully.")

# ==========================================================
# SWEETVIZ
# ==========================================================

print("\n========== SWEETVIZ ==========")

analysis = sv.analyze(df)

analysis.show_html(

    "reports/titanic_sweetviz.html"

)

print("Sweetviz Report Saved Successfully.")

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)

print("""

Dataset Name
------------
Titanic

Number of Records
-----------------
{}

Number of Features
------------------
{}

""".format(df.shape[0], df.shape[1]))

# ==========================================================
# KEY FINDINGS
# ==========================================================

print("\n========== KEY FINDINGS ==========")

print("""

1.

Age contains missing values.

2.

Deck contains a very large number
of missing values.

3.

Passenger Fare is highly right-skewed.

4.

Survival is influenced by

• Gender

• Passenger Class

• Fare

5.

The dataset contains both numerical
and categorical features.

""")

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

print("\n========== BUSINESS INSIGHTS ==========")

print("""

Insight 1
---------
Female passengers have a considerably
higher survival rate.

Insight 2
---------
Passengers travelling in First Class
survived more frequently.

Insight 3
---------
Higher ticket fare is associated with
greater survival probability.

Insight 4
---------
Passenger Class should be retained
during feature engineering.

""")

# ==========================================================
# RECOMMENDED PREPROCESSING
# ==========================================================

print("\n========== RECOMMENDED PREPROCESSING ==========")

print("""

✓ Fill missing values

✓ Remove unnecessary columns

✓ Encode categorical variables

✓ Scale numerical features

✓ Detect outliers

✓ Perform feature engineering

✓ Split Train/Test

✓ Train Machine Learning model

""")

# ==========================================================
# MANUAL VS AUTOMATED EDA
# ==========================================================

print("\n========== COMPARISON ==========")

comparison = pd.DataFrame({

    "Manual EDA":[

        "More Flexible",

        "Custom Visualizations",

        "Business Interpretation",

        "Time Consuming"

    ],

    "Automated EDA":[

        "Very Fast",

        "Comprehensive Reports",

        "Limited Customization",

        "Ideal for Initial Exploration"

    ]

})

print(comparison)

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n========== MINI PRACTICE ==========")

print("""

1.

Generate an automated report for
the Iris Dataset.

2.

Generate a report for the
Tips Dataset.

3.

Compare Manual EDA and
Automated EDA.

4.

Write five business insights.

5.

Suggest preprocessing steps
before Machine Learning.

""")

# ==========================================================
# LESSON CONCLUSION
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""

Skills Learned

✔ Automated EDA

✔ HTML Report Generation

✔ Data Quality Assessment

✔ Business Storytelling

✔ Executive Summary Writing

✔ Data Interpretation

✔ ML Preparation

""")

# ==========================================================
# LESSON COMPLETED
# ==========================================================

print("\n" + "=" * 80)
print("LESSON 07 COMPLETED SUCCESSFULLY")
print("=" * 80)
