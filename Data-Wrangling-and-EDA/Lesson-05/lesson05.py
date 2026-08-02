"""
Lesson 05 : Feature Engineering

Topics Covered
---------------
1. Label Encoding
2. One-Hot Encoding
3. Frequency Encoding
4. Min-Max Scaling
5. Standard Scaling
6. Robust Scaling
7. Binning
8. Polynomial Features
9. Interaction Features
10. Domain Feature Engineering
11. SMOTE
12. ADASYN

Author : Krish Goel
Repository : Summer Training
"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PolynomialFeatures
)

from imblearn.over_sampling import SMOTE, ADASYN

# ==================================================
# LOAD DATASET
# ==================================================

print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

df = sns.load_dataset("titanic")

print("\nOriginal Dataset")
print(df.head())

# ==================================================
# LABEL ENCODING
# ==================================================
# Converts categories into integer labels.

print("\n========== LABEL ENCODING ==========")

label = LabelEncoder()

df["sex_encoded"] = label.fit_transform(df["sex"])

print(df[["sex", "sex_encoded"]].head())

# ==================================================
# ONE-HOT ENCODING
# ==================================================
# Creates binary columns for each category.

print("\n========== ONE-HOT ENCODING ==========")

embarked_encoded = pd.get_dummies(
    df["embarked"],
    prefix="Embarked"
)

print(embarked_encoded.head())

# ==================================================
# FREQUENCY ENCODING
# ==================================================
# Replace category with occurrence frequency.

print("\n========== FREQUENCY ENCODING ==========")

frequency = df["class"].value_counts()

df["class_frequency"] = df["class"].map(frequency)

print(df[["class", "class_frequency"]].head())

# ==================================================
# FEATURE SCALING
# ==================================================

numeric = df[["age", "fare"]].fillna(df[["age", "fare"]].median())

# ---------------- Min-Max ----------------

print("\n========== MIN-MAX SCALING ==========")

minmax = MinMaxScaler()

print(minmax.fit_transform(numeric)[:5])

# ---------------- Standard ----------------

print("\n========== STANDARD SCALING ==========")

standard = StandardScaler()

print(standard.fit_transform(numeric)[:5])

# ---------------- Robust ----------------

print("\n========== ROBUST SCALING ==========")

robust = RobustScaler()

print(robust.fit_transform(numeric)[:5])

# ==================================================
# BINNING
# ==================================================
# Equal Width

print("\n========== BINNING ==========")

df["Age_Group"] = pd.cut(
    df["age"],
    bins=5
)

print(df["Age_Group"].head())

# Equal Frequency

df["Fare_Group"] = pd.qcut(
    df["fare"],
    q=4,
    duplicates="drop"
)

print(df["Fare_Group"].head())

# ==================================================
# POLYNOMIAL FEATURES
# ==================================================

print("\n========== POLYNOMIAL FEATURES ==========")

poly = PolynomialFeatures(degree=2, include_bias=False)

poly_features = poly.fit_transform(numeric)

print(poly_features[:3])

# ==================================================
# DOMAIN FEATURE ENGINEERING
# ==================================================
# Create a business/domain-specific feature.

print("\n========== DOMAIN FEATURE ==========")

df["Family_Size"] = df["sibsp"] + df["parch"] + 1

print(df[["sibsp", "parch", "Family_Size"]].head())

# ==================================================
# IMBALANCED DATA
# ==================================================
# Demonstration only.

print("\n========== SMOTE ==========")

X = numeric
y = df["survived"]

smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, y)

print("Original Samples :", len(y))
print("After SMOTE :", len(y_smote))

print("\n========== ADASYN ==========")

adasyn = ADASYN(random_state=42)

X_ada, y_ada = adasyn.fit_resample(X, y)

print("After ADASYN :", len(y_ada))

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

print("Average Fare")

print(df["fare"].mean())

print("\nUnique Embarked Values")

print(df["embarked"].unique())

print("\nFeature Engineering Completed Successfully!")
