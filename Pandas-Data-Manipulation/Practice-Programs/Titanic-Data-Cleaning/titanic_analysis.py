"""
Project : Titanic Data Cleaning

Description:
------------
Perform basic data cleaning and exploration
using the Titanic dataset.

Concepts Used:
--------------
- Data Loading
- Missing Values
- Boolean Masking
- Data Cleaning
"""

import pandas as pd
import seaborn as sns

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

titanic = sns.load_dataset("titanic")

print("=" * 60)
print("TITANIC DATASET")
print("=" * 60)

print("\nFirst Five Rows")
print(titanic.head())

print("\nDataset Information")
print(titanic.info())

print("\nMissing Values")
print(titanic.isnull().sum())

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

cleaned = titanic.copy()

cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())

cleaned = cleaned.dropna(subset=["embarked"])

cleaned = cleaned.rename(columns={"survived": "Survived"})

cleaned["Survived"] = cleaned["Survived"].astype(int)

print("\nCleaned Dataset")
print(cleaned.head())

print("\nPassengers Older Than 30")
print(cleaned[cleaned["age"] > 30][["age", "sex", "class"]])
