"""
Lesson 02 : Data Manipulation & Time Series

Topics Covered
---------------
1. GroupBy
2. Aggregation
3. Pivot Tables
4. Crosstab
5. Merge
6. Join
7. Concat
8. Apply
9. Map
10. ApplyMap
11. Time Series

Author : Krish Goel
Repository : Summer Training
"""

import pandas as pd
import numpy as np

# ==================================================
# SAMPLE DATASET
# ==================================================

employees = pd.DataFrame({
    "Employee": ["A", "B", "C", "D", "E"],
    "Department": ["IT", "HR", "IT", "Finance", "HR"],
    "Salary": [60000, 45000, 70000, 80000, 50000],
    "Experience": [2, 5, 4, 7, 3]
})

print("\n========== ORIGINAL DATA ==========")
print(employees)

# ==================================================
# GROUPBY
# ==================================================
# Groups rows having the same value.

print("\n========== GROUPBY ==========")

department_salary = employees.groupby("Department")["Salary"].mean()

print(department_salary)

# ==================================================
# AGGREGATION
# ==================================================
# Perform multiple aggregation functions.

print("\n========== AGGREGATION ==========")

aggregation = employees.groupby("Department").agg({
    "Salary": ["mean", "max", "min"],
    "Experience": "mean"
})

print(aggregation)

# ==================================================
# PIVOT TABLE
# ==================================================
# Creates a summarized table.

print("\n========== PIVOT TABLE ==========")

pivot = pd.pivot_table(
    employees,
    values="Salary",
    index="Department",
    aggfunc="mean"
)

print(pivot)

# ==================================================
# CROSSTAB
# ==================================================
# Generates a frequency table.

print("\n========== CROSSTAB ==========")

employees["Bonus"] = ["Yes", "No", "Yes", "Yes", "No"]

cross = pd.crosstab(
    employees["Department"],
    employees["Bonus"]
)

print(cross)

# ==================================================
# MERGE
# ==================================================
# Combines DataFrames using common columns.

print("\n========== MERGE ==========")

employee_info = pd.DataFrame({
    "Employee": ["A", "B", "C", "D"],
    "City": ["Delhi", "Mumbai", "Pune", "Jaipur"]
})

merged = pd.merge(
    employees,
    employee_info,
    on="Employee",
    how="inner"
)

print(merged)

# ==================================================
# LEFT / RIGHT / OUTER JOIN
# ==================================================

print("\n========== LEFT JOIN ==========")
print(pd.merge(employees, employee_info, on="Employee", how="left"))

print("\n========== RIGHT JOIN ==========")
print(pd.merge(employees, employee_info, on="Employee", how="right"))

print("\n========== OUTER JOIN ==========")
print(pd.merge(employees, employee_info, on="Employee", how="outer"))

# ==================================================
# JOIN
# ==================================================
# Joins DataFrames using index.

print("\n========== JOIN ==========")

df1 = employees.set_index("Employee")
df2 = employee_info.set_index("Employee")

print(df1.join(df2))

# ==================================================
# CONCAT
# ==================================================
# Concatenates DataFrames vertically or horizontally.

print("\n========== CONCAT ==========")

first = pd.DataFrame({"A": [1, 2]})
second = pd.DataFrame({"A": [3, 4]})

print(pd.concat([first, second], ignore_index=True))

# ==================================================
# APPLY
# ==================================================
# Applies a function to a Series or DataFrame.

print("\n========== APPLY ==========")

employees["Salary_in_Lakhs"] = employees["Salary"].apply(lambda x: x / 100000)

print(employees)

# ==================================================
# MAP
# ==================================================
# Maps values in a Series.

print("\n========== MAP ==========")

grade = {
    "IT": "Technical",
    "HR": "Management",
    "Finance": "Accounts"
}

employees["Category"] = employees["Department"].map(grade)

print(employees)

# ==================================================
# APPLYMAP
# ==================================================
# Applies a function element-wise on an entire DataFrame.

print("\n========== APPLYMAP ==========")

numbers = pd.DataFrame({
    "A": [1, 2],
    "B": [3, 4]
})

print(numbers.applymap(lambda x: x * 10))

# ==================================================
# TIME SERIES
# ==================================================
# Pandas provides powerful tools for date and time data.

print("\n========== TIME SERIES ==========")

dates = pd.date_range(
    start="2026-01-01",
    periods=10,
    freq="D"
)

sales = pd.DataFrame({
    "Sales": [100, 120, 140, 150, 170, 180, 200, 210, 220, 250]
}, index=dates)

print(sales)

# --------------------------------------------------
# RESAMPLE
# --------------------------------------------------
# Converts daily data into larger time intervals.

print("\nMonthly Resample")

print(sales.resample("M").sum())

# --------------------------------------------------
# ROLLING
# --------------------------------------------------
# Calculates moving statistics.

print("\nRolling Average")

print(sales.rolling(3).mean())

# --------------------------------------------------
# SHIFT
# --------------------------------------------------
# Shifts values by a specified number of periods.

print("\nShifted Sales")

sales["Previous_Day_Sales"] = sales["Sales"].shift(1)

print(sales)

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

students = pd.DataFrame({
    "Name": ["Krish", "Adi", "Varun", "Rahul"],
    "Marks": [92, 88, 75, 95],
    "Section": ["A", "A", "B", "B"]
})

print("\nAverage Marks by Section")

print(students.groupby("Section")["Marks"].mean())

students["Grade"] = students["Marks"].apply(
    lambda x: "Pass" if x >= 80 else "Needs Improvement"
)

print("\nStudent Grades")

print(students)

# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 02 Completed Successfully!")
