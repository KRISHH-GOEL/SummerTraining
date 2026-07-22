"""
Lesson 01 : Pandas Basics, Data Loading & Data Cleaning

Topics Covered
---------------
1. Series
2. DataFrame
3. DataFrame Attributes
4. Loading Data
5. Data Selection
6. Boolean Masking
7. Data Cleaning

Author : Krish Goel
Repository : Summer Training
"""

import pandas as pd
import numpy as np

# ==================================================
# INTRODUCTION TO PANDAS
# ==================================================

print("\n========== PANDAS BASICS ==========")

# --------------------------------------------------
# SERIES
# --------------------------------------------------
# A Series is a one-dimensional labeled array.
# It can store integers, floats, strings, etc.

series = pd.Series([10, 20, 30, 40, 50])

print("\nSeries")

print(series)

print("\nSeries Data Type")

print(series.dtype)

# --------------------------------------------------
# DATAFRAME
# --------------------------------------------------
# A DataFrame is a two-dimensional table with
# labeled rows and columns.

student_df = pd.DataFrame(
    {
        "Name": ["Krish", "Adi", "Varun", "Rahul"],
        "Age": [20, 21, 20, 22],
        "Marks": [92, 88, 75, 95],
    }
)

print("\nDataFrame")

print(student_df)

# --------------------------------------------------
# DATAFRAME ATTRIBUTES
# --------------------------------------------------

print("\nShape")

print(student_df.shape)

print("\nColumns")

print(student_df.columns)

print("\nIndex")

print(student_df.index)

print("\nData Types")

print(student_df.dtypes)

# ==================================================
# LOADING DATA
# ==================================================

print("\n========== LOADING DATA ==========")

# These are example syntaxes.
# Uncomment when actual files are available.

# CSV File
# df_csv = pd.read_csv("students.csv")

# Excel File
# df_excel = pd.read_excel("students.xlsx")

# JSON File
# df_json = pd.read_json("students.json")

# SQL Database
# df_sql = pd.read_sql("SELECT * FROM students", connection)

print("Examples of reading CSV, Excel, JSON, and SQL are shown as comments.")

# ==================================================
# DATA SELECTION
# ==================================================

print("\n========== DATA SELECTION ==========")

print("\nUsing loc[] (Label-based Selection)")

print(student_df.loc[0])

print("\nUsing iloc[] (Position-based Selection)")

print(student_df.iloc[1])

print("\nSelecting Multiple Columns")

print(student_df[["Name", "Marks"]])

# --------------------------------------------------
# BOOLEAN MASKING
# --------------------------------------------------
# Used to filter rows based on conditions.

print("\nStudents Scoring Above 85")

print(student_df[student_df["Marks"] > 85])

# ==================================================
# DATA CLEANING
# ==================================================

print("\n========== DATA CLEANING ==========")

dirty_df = pd.DataFrame(
    {
        "Name": ["Krish", "Adi", "Adi", np.nan],
        "Age": [20, 21, 21, 22],
        "Marks": [90, np.nan, np.nan, 80],
    }
)

print("\nOriginal DataFrame")

print(dirty_df)

# --------------------------------------------------
# dropna()
# --------------------------------------------------
# Removes rows containing missing values.

print("\nUsing dropna()")

print(dirty_df.dropna())

# --------------------------------------------------
# fillna()
# --------------------------------------------------
# Replaces missing values.

print("\nUsing fillna()")

filled_df = dirty_df.fillna(0)

print(filled_df)

# --------------------------------------------------
# drop_duplicates()
# --------------------------------------------------
# Removes duplicate rows.

print("\nUsing drop_duplicates()")

print(dirty_df.drop_duplicates())

# --------------------------------------------------
# astype()
# --------------------------------------------------
# Changes data types.

filled_df["Age"] = filled_df["Age"].astype(int)

print("\nUsing astype()")

print(filled_df.dtypes)

# --------------------------------------------------
# rename()
# --------------------------------------------------
# Renames column names.

renamed_df = filled_df.rename(
    columns={
        "Marks": "Score"
    }
)

print("\nUsing rename()")

print(renamed_df)

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

employees = pd.DataFrame(
    {
        "Employee": ["A", "B", "C", "D"],
        "Salary": [45000, 55000, 70000, 65000],
        "Department": ["HR", "IT", "Finance", "IT"],
    }
)

print("\nEmployee Data")

print(employees)

print("\nEmployees with Salary > 50000")

print(employees[employees["Salary"] > 50000])

print("\nEmployee Names Only")

print(employees["Employee"])

# Rename Department column

employees = employees.rename(columns={"Department": "Dept"})

print("\nUpdated DataFrame")

print(employees)

# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 01 Completed Successfully!")
