"""
Lesson 03 : Data Cleaning & Data Preprocessing

Topics Covered
---------------
1. Missing Values
2. Missing Value Imputation
3. Duplicate Records
4. Structural Errors
5. DateTime Parsing
6. Schema Validation

Author : Krish Goel
Repository : Summer Training
"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# ==================================================
# LOAD DATASET
# ==================================================

print("=" * 70)
print("DATA CLEANING & PREPROCESSING")
print("=" * 70)

# Titanic Dataset
df = sns.load_dataset("titanic")

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ==================================================
# TYPES OF MISSING VALUES
# ==================================================

print("\n========== MISSING VALUES ==========")

print("""
MCAR
- Missing Completely At Random

MAR
- Missing At Random

MNAR
- Missing Not At Random
""")

# ==================================================
# SIMPLE IMPUTATION
# ==================================================

print("\n========== SIMPLE IMPUTATION ==========")

age_df = df[["age"]]

mean_imputer = SimpleImputer(strategy="mean")

age_df["age"] = mean_imputer.fit_transform(age_df)

print(age_df.head())

# ==================================================
# MEDIAN IMPUTATION
# ==================================================

median_imputer = SimpleImputer(strategy="median")

age_df["age"] = median_imputer.fit_transform(age_df)

# ==================================================
# MOST FREQUENT (MODE)
# ==================================================

embarked_df = df[["embarked"]]

mode_imputer = SimpleImputer(strategy="most_frequent")

embarked_df["embarked"] = mode_imputer.fit_transform(embarked_df)

# ==================================================
# KNN IMPUTATION
# ==================================================

print("\n========== KNN IMPUTATION ==========")

numeric = df.select_dtypes(include=np.number)

knn = KNNImputer(n_neighbors=5)

filled = knn.fit_transform(numeric)

filled = pd.DataFrame(filled, columns=numeric.columns)

print(filled.head())

# ==================================================
# DUPLICATE RECORDS
# ==================================================

print("\n========== DUPLICATES ==========")

print("Duplicate Rows :", df.duplicated().sum())

df = df.drop_duplicates()

print("After Removing :", df.duplicated().sum())

# ==================================================
# STRUCTURAL ERRORS
# ==================================================

print("\n========== STRUCTURAL ERRORS ==========")

sample = pd.DataFrame({

    "City": [" Delhi ", "MUMBAI", " pune ", "Delhi"],

})

print("\nOriginal")

print(sample)

sample["City"] = sample["City"].str.strip()

sample["City"] = sample["City"].str.lower()

sample["City"] = sample["City"].str.title()

print("\nCleaned")

print(sample)

# ==================================================
# DATETIME PARSING
# ==================================================

print("\n========== DATETIME ==========")

orders = pd.DataFrame({

    "Order_Date":[

        "2026-01-05",

        "2026-02-18",

        "2026-03-22"

    ]

})

orders["Order_Date"] = pd.to_datetime(orders["Order_Date"])

orders["Year"] = orders["Order_Date"].dt.year

orders["Month"] = orders["Order_Date"].dt.month

orders["Day"] = orders["Order_Date"].dt.day

orders["Day_Name"] = orders["Order_Date"].dt.day_name()

orders["Quarter"] = orders["Order_Date"].dt.quarter

print(orders)

# ==================================================
# SCHEMA VALIDATION
# ==================================================

print("\n========== SCHEMA VALIDATION ==========")

print(df.dtypes)

df["survived"] = df["survived"].astype(int)

print("\nUpdated Data Types")

print(df.dtypes)

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

print("Passengers older than 60")

print(df[df["age"] > 60][["age", "sex", "class"]])

print("\nAverage Passenger Age")

print(df["age"].mean())

print("\nUnique Embarkation Ports")

print(df["embarked"].unique())

# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 03 Completed Successfully!")
