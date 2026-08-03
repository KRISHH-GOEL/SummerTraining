"""
Lesson 06 : Exploratory Data Analysis (EDA)

Topics Covered
---------------
1. Dataset Overview
2. Data Inspection
3. Missing Value Analysis
4. Duplicate Detection
5. Unique Value Analysis
6. Statistical Summary
7. Univariate Analysis
    - Histogram
    - KDE Plot
    - Count Plot

Dataset Used
-------------
Titanic Dataset (Seaborn)

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Create folder to save plots
# ----------------------------------------------------------

os.makedirs("images", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Load Titanic Dataset

df = sns.load_dataset("titanic")

print("\nDataset Loaded Successfully.")

# ==========================================================
# BASIC DATA INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print("\nFirst Five Records")
print(df.head())

print("\nLast Five Records")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

df.info()

# ==========================================================
# STATISTICAL SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

print(df.describe())

print("\nCategorical Summary")

print(df.describe(include="object"))

# ==========================================================
# MISSING VALUE ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS")
print("=" * 80)

missing = df.isnull().sum()

missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({

    "Missing Values": missing,

    "Percentage": missing_percent.round(2)

})

print(missing_df.sort_values(by="Percentage", ascending=False))

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Business Insight
----------------
• deck contains a large number of missing values.

• age has moderate missing values.

• embarked and embark_town contain only a few
  missing observations.

Cleaning Strategy

1. Drop deck column if required.

2. Impute age using median.

3. Fill embarked using mode.

""")

# ==========================================================
# DUPLICATE RECORDS
# ==========================================================

print("\n" + "=" * 80)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 80)

duplicates = df.duplicated().sum()

print("Duplicate Records :", duplicates)

# ==========================================================
# UNIQUE VALUE ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("UNIQUE VALUES")
print("=" * 80)

for column in df.columns:

    print(f"\n{column}")

    print("Unique Values :", df[column].nunique())

# ==========================================================
# UNIVARIATE ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("UNIVARIATE ANALYSIS")
print("=" * 80)

# ----------------------------------------------------------
# HISTOGRAM
# ----------------------------------------------------------
# Histogram helps understand the distribution
# of numerical variables.

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="age",
    bins=20,
    kde=False
)

plt.title("Distribution of Passenger Age")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("images/age_histogram.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Histogram Insight
-----------------
• Most passengers are between 20 and 40 years old.

• Very young and very old passengers are fewer
  in number.

• Age is not uniformly distributed.

""")

# ==========================================================
# KDE PLOT
# ==========================================================

plt.figure(figsize=(8,5))

sns.kdeplot(

    data=df,

    x="fare",

    fill=True

)

plt.title("Fare Distribution (KDE)")

plt.tight_layout()

plt.savefig("images/fare_kde.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

KDE Insight
-----------
• Fare distribution is highly right-skewed.

• Most passengers paid relatively low fares.

• A few passengers paid extremely high fares.

Possible Action
---------------
Log Transformation may improve model performance.

""")

# ==========================================================
# COUNT PLOT
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(

    data=df,

    x="survived"

)

plt.title("Passenger Survival Count")

plt.tight_layout()

plt.savefig("images/survival_count.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

survival_rate = (
    df["survived"].value_counts(normalize=True) * 100
)

print("""

Survival Insight
----------------
""")

print(survival_rate)

print("""

Observation

• The dataset is moderately imbalanced.

• More passengers did not survive than survived.

• Accuracy alone may not be an appropriate metric
  for future Machine Learning models.

Recommendation

Use Precision, Recall,
F1-Score and ROC-AUC
during model evaluation.

""")

# ==========================================================
# GENDER DISTRIBUTION
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(

    data=df,

    x="sex"

)

plt.title("Gender Distribution")

plt.tight_layout()

plt.savefig("images/gender_distribution.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

gender_percent = (
    df["sex"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("""

Gender Distribution

""")

print(gender_percent)

print("""

Observation

• Male passengers outnumber female passengers.

• This imbalance should be considered when
  interpreting survival statistics.

""")
# ==========================================================
# PIE CHART
# ==========================================================
# Pie charts show the proportion of categories.

survival_counts = df["survived"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    survival_counts,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Passenger Survival Percentage")

plt.savefig("images/survival_pie.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Pie Chart Insight
-----------------
• Around two-thirds of the passengers did not survive.

• The target variable is slightly imbalanced.

""")

# ==========================================================
# PASSENGER CLASS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="class"
)

plt.title("Passenger Class Distribution")

plt.tight_layout()

plt.savefig("images/passenger_class_distribution.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Passenger Class Insight
-----------------------
• Third Class has the highest number of passengers.

• First Class has the fewest passengers.

""")

# ==========================================================
# BOXPLOT
# ==========================================================
# Detect outliers in Fare.

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    y="fare"
)

plt.title("Fare Distribution")

plt.tight_layout()

plt.savefig("images/fare_boxplot.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Boxplot Insight
---------------
• Fare contains several extreme outliers.

• A few passengers paid significantly higher fares.

Possible Action
---------------
Consider:
- Log Transformation
- Winsorization
- Robust Scaling

""")

# ==========================================================
# AGE DISTRIBUTION BY PASSENGER CLASS
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="class",
    y="age"
)

plt.title("Age Distribution Across Passenger Classes")

plt.tight_layout()

plt.savefig("images/age_class_boxplot.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Observation
-----------
• First-class passengers generally have a higher median age.

• Third-class passengers contain a larger number of younger passengers.

""")

# ==========================================================
# SCATTER PLOT
# ==========================================================
# Relationship between Age and Fare.

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title("Age vs Fare")

plt.tight_layout()

plt.savefig("images/age_fare_scatter.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Scatter Plot Insight
--------------------
• Higher fares are generally associated with older passengers.

• Survivors are concentrated among passengers
  paying relatively higher fares.

""")

# ==========================================================
# VIOLIN PLOT
# ==========================================================
# Distribution + Density.

plt.figure(figsize=(8,5))

sns.violinplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival")

plt.tight_layout()

plt.savefig("images/violin_survival.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Violin Plot Insight
-------------------
• Survivors tend to have a slightly younger age distribution.

• Children appear to have relatively higher survival rates.

""")

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

print("\n" + "=" * 80)
print("CORRELATION MATRIX")
print("=" * 80)

correlation = df.corr(numeric_only=True)

print(correlation)

# ==========================================================
# HEATMAP
# ==========================================================

plt.figure(figsize=(10,7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("images/correlation_heatmap.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Heatmap Insight
---------------
• Fare has a positive correlation with survival.

• Passenger Class has a negative relationship with survival.

• Age shows only a weak relationship.

""")

# ==========================================================
# PAIRPLOT
# ==========================================================

sns.pairplot(
    df[
        [
            "age",
            "fare",
            "survived",
            "pclass"
        ]
    ].dropna(),
    hue="survived"
)

plt.savefig("images/pairplot.png")

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Pair Plot Insight
-----------------
• Fare is one of the strongest distinguishing features.

• Age alone cannot clearly separate survivors
  from non-survivors.

""")

# ==========================================================
# FEATURE RELATIONSHIP SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE RELATIONSHIP SUMMARY")
print("=" * 80)

print("""

Important Observations
----------------------

✓ Fare is positively associated with survival.

✓ Passenger Class influences survival probability.

✓ Age has weak correlation.

✓ Fare contains outliers.

✓ Age contains missing values.

✓ Survival dataset is slightly imbalanced.

These observations will help during:

• Feature Engineering
• Data Cleaning
• Feature Selection
• Machine Learning

""")
# ==========================================================
# CLASS-WISE SURVIVAL ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("CLASS-WISE SURVIVAL ANALYSIS")
print("=" * 80)

class_survival = (
    df.groupby("class")["survived"]
    .mean()
    .sort_values(ascending=False)
)

print(class_survival)

plt.figure(figsize=(8,5))

class_survival.plot(
    kind="bar"
)

plt.title("Average Survival Rate by Passenger Class")

plt.xlabel("Passenger Class")

plt.ylabel("Average Survival Rate")

plt.tight_layout()

plt.savefig("images/class_survival_rate.png")

plt.show()

print("""

Business Insight
----------------

• First Class passengers have the highest survival rate.

• Third Class passengers have the lowest survival rate.

• Passenger Class appears to be one of the
  strongest predictors of survival.

""")

# ==========================================================
# GENDER-WISE SURVIVAL
# ==========================================================

print("\n" + "=" * 80)
print("GENDER-WISE SURVIVAL")
print("=" * 80)

gender_survival = (
    df.groupby("sex")["survived"]
    .mean()
)

print(gender_survival)

plt.figure(figsize=(6,5))

gender_survival.plot(
    kind="bar"
)

plt.title("Average Survival Rate by Gender")

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig("images/gender_survival.png")

plt.show()

print("""

Business Insight
----------------

• Female passengers survived at a much higher rate.

• Gender is likely to be an important feature
  for prediction models.

""")

# ==========================================================
# EMBARKATION ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("EMBARKATION ANALYSIS")
print("=" * 80)

embarked = (
    df.groupby("embarked")["survived"]
    .mean()
)

print(embarked)

plt.figure(figsize=(7,5))

embarked.plot(
    kind="bar"
)

plt.title("Average Survival by Embarkation Port")

plt.tight_layout()

plt.savefig("images/embarked_survival.png")

plt.show()

print("""

Business Insight
----------------

Passengers embarking from different ports
show different survival rates.

This suggests that embarkation location
may carry predictive information.

""")

# ==========================================================
# DATA QUALITY REPORT
# ==========================================================

print("\n" + "=" * 80)
print("DATA QUALITY REPORT")
print("=" * 80)

print(f"""
Total Records           : {len(df)}

Total Features          : {df.shape[1]}

Duplicate Rows          : {df.duplicated().sum()}

Missing Values Present  : {df.isnull().sum().sum()}

Numerical Features      : {len(df.select_dtypes(include='number').columns)}

Categorical Features    : {len(df.select_dtypes(include='object').columns)}

""")

# ==========================================================
# FEATURE IMPORTANCE OBSERVATIONS
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE OBSERVATIONS")
print("=" * 80)

print("""

Important Features
------------------

✔ Passenger Class

✔ Fare

✔ Sex

✔ Age

Moderately Useful Features
--------------------------

✔ Embarked

✔ SibSp

✔ Parch

Low Priority Features
---------------------

✔ Cabin (Too many missing values)

✔ Deck (Large percentage of missing values)

""")

# ==========================================================
# MACHINE LEARNING RECOMMENDATIONS
# ==========================================================

print("\n" + "=" * 80)
print("RECOMMENDATIONS BEFORE MACHINE LEARNING")
print("=" * 80)

print("""

Recommended Preprocessing Pipeline

1. Remove unnecessary columns.

2. Handle missing values.

3. Encode categorical variables.

4. Scale numerical features.

5. Detect and treat outliers.

6. Split into Train/Test.

7. Train Machine Learning model.

""")

# ==========================================================
# FINAL EDA SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("FINAL EDA SUMMARY")
print("=" * 80)

summary = {

    "Rows": df.shape[0],

    "Columns": df.shape[1],

    "Missing Values": int(df.isnull().sum().sum()),

    "Duplicate Rows": int(df.duplicated().sum()),

    "Numerical Columns":
        len(df.select_dtypes(include="number").columns),

    "Categorical Columns":
        len(df.select_dtypes(include="object").columns)

}

for key, value in summary.items():

    print(f"{key:<25}: {value}")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""

Try Performing the Following Exercises

1.
Find the average fare paid by males
and females separately.

2.
Find the oldest passenger.

3.
Find the youngest passenger.

4.
Find passengers travelling alone.

5.
Find passengers having family size > 4.

6.
Find survival percentage for each class.

7.
Create a histogram of Fare.

8.
Create a heatmap using only selected
numerical features.

9.
Find the top five highest fares.

10.
Write three business insights from
the Titanic dataset.

""")

# ==========================================================
# CONCLUSION
# ==========================================================

print("\n" + "=" * 80)
print("LESSON CONCLUSION")
print("=" * 80)

print("""

This lesson demonstrated the complete workflow
of Exploratory Data Analysis (EDA).

Key Skills Learned

✔ Dataset Inspection

✔ Data Quality Assessment

✔ Missing Value Analysis

✔ Univariate Analysis

✔ Bivariate Analysis

✔ Multivariate Analysis

✔ Business Storytelling

✔ Data-driven Decision Making

These skills form the foundation of every
successful Data Science and Machine Learning project.

""")

# ==========================================================
# LESSON COMPLETED
# ==========================================================

print("\n" + "=" * 80)
print("LESSON 06 COMPLETED SUCCESSFULLY")
print("=" * 80)
