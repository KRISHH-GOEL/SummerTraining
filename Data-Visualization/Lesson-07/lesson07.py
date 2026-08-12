
# `python_data_preparation.py`

```python
"""
Lesson 07 : Power BI for Data Science

Python Data Preparation Component

Topics Covered
---------------
1. Data Creation
2. Data Inspection
3. Data Cleaning
4. Data Type Conversion
5. Missing Value Handling
6. Duplicate Removal
7. Feature Engineering
8. Aggregation
9. Exporting Data for Power BI

Purpose
-------
This script demonstrates how Python can be used to
prepare data before importing it into Power BI.

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import numpy as np
import pandas as pd

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs("sample_data", exist_ok=True)

# ==========================================================
# CREATE SAMPLE SALES DATA
# ==========================================================

print("=" * 80)
print("POWER BI - PYTHON DATA PREPARATION")
print("=" * 80)

data = {

    "OrderID": [
        1001, 1002, 1003, 1004, 1005,
        1006, 1007, 1008, 1009, 1010,
        1011, 1012
    ],

    "Date": [
        "2026-01-05",
        "2026-01-08",
        "2026-01-15",
        "2026-02-03",
        "2026-02-11",
        "2026-02-20",
        "2026-03-02",
        "2026-03-10",
        "2026-03-18",
        "2026-04-01",
        "2026-04-12",
        "2026-04-25"
    ],

    "Product": [
        "Laptop",
        "Mouse",
        "Keyboard",
        "Laptop",
        "Monitor",
        "Mouse",
        "Keyboard",
        "Laptop",
        "Monitor",
        "Mouse",
        "Laptop",
        "Keyboard"
    ],

    "Category": [
        "Electronics",
        "Accessories",
        "Accessories",
        "Electronics",
        "Electronics",
        "Accessories",
        "Accessories",
        "Electronics",
        "Electronics",
        "Accessories",
        "Electronics",
        "Accessories"
    ],

    "Region": [
        "North",
        "West",
        "North",
        "South",
        "West",
        "East",
        "North",
        "South",
        "West",
        "East",
        "North",
        "South"
    ],

    "Quantity": [
        2, 5, 3, 1, 2, 8,
        4, 1, 3, 6, 2, 5
    ],

    "Sales": [
        120000,
        5000,
        7500,
        60000,
        30000,
        8000,
        10000,
        65000,
        45000,
        6000,
        130000,
        12500
    ],

    "Profit": [
        18000,
        1500,
        1800,
        9000,
        7500,
        2400,
        3000,
        10000,
        9000,
        1800,
        19500,
        3750
    ]
}

df = pd.DataFrame(data)

print("\nOriginal Dataset")

print(df)

# ==========================================================
# DATA INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("DATA INSPECTION")
print("=" * 80)

print("\nShape")

print(df.shape)

print("\nData Types")

print(df.dtypes)

print("\nMissing Values")

print(df.isnull().sum())

print("\nDuplicate Rows")

print(df.duplicated().sum())

# ==========================================================
# DATE CONVERSION
# ==========================================================

print("\n" + "=" * 80)
print("DATE CONVERSION")
print("=" * 80)

df["Date"] = pd.to_datetime(
    df["Date"]
)

print(df["Date"].dtype)

# ==========================================================
# DATE FEATURE ENGINEERING
# ==========================================================

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Month_Name"] = df["Date"].dt.month_name()

df["Quarter"] = (
    "Q"
    + df["Date"].dt.quarter.astype(str)
)

df["Day"] = df["Date"].dt.day

df["Day_Name"] = (
    df["Date"].dt.day_name()
)

# ==========================================================
# PROFIT MARGIN
# ==========================================================

df["Profit_Margin"] = (
    df["Profit"]
    / df["Sales"]
    * 100
)

# ==========================================================
# AVERAGE SELLING VALUE
# ==========================================================

df["Average_Order_Value"] = (
    df["Sales"]
    / df["Quantity"]
)

print("\nFeature Engineering Completed")

print(
    df[
        [
            "Date",
            "Year",
            "Month",
            "Quarter",
            "Profit_Margin",
            "Average_Order_Value"
        ]
    ]
)

# ==========================================================
# DATA TYPE ENFORCEMENT
# ==========================================================

print("\n" + "=" * 80)
print("DATA TYPE ENFORCEMENT")
print("=" * 80)

df["OrderID"] = df["OrderID"].astype(int)

df["Quantity"] = df["Quantity"].astype(int)

df["Sales"] = df["Sales"].astype(float)

df["Profit"] = df["Profit"].astype(float)

# ==========================================================
# DATA QUALITY CHECK
# ==========================================================

print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

print("\nMissing Values")

print(df.isnull().sum())

print("\nDuplicates")

print(df.duplicated().sum())

print("\nData Types")

print(df.dtypes)

# ==========================================================
# BUSINESS AGGREGATION
# ==========================================================

print("\n" + "=" * 80)
print("REGION-WISE SALES")
print("=" * 80)

region_summary = (
    df.groupby("Region")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Total_Orders=("OrderID", "count")
    )
    .reset_index()
)

print(region_summary)

# ==========================================================
# CATEGORY SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("CATEGORY-WISE PERFORMANCE")
print("=" * 80)

category_summary = (
    df.groupby("Category")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum")
    )
    .reset_index()
)

print(category_summary)

# ==========================================================
# MONTHLY SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("MONTHLY SALES")
print("=" * 80)

monthly_summary = (
    df.groupby(
        ["Year", "Month", "Month_Name"],
        as_index=False
    )
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum")
    )
)

print(monthly_summary)

# ==========================================================
# POWER BI KPI VALUES
# ==========================================================

print("\n" + "=" * 80)
print("POWER BI KPI VALUES")
print("=" * 80)

total_sales = df["Sales"].sum()

total_profit = df["Profit"].sum()

total_quantity = df["Quantity"].sum()

total_orders = df["OrderID"].nunique()

average_order_value = (
    df["Sales"].sum()
    / df["OrderID"].nunique()
)

profit_margin = (
    df["Profit"].sum()
    / df["Sales"].sum()
    * 100
)

print(
    f"Total Sales       : ₹{total_sales:,.2f}"
)

print(
    f"Total Profit      : ₹{total_profit:,.2f}"
)

print(
    f"Total Quantity    : {total_quantity:,}"
)

print(
    f"Total Orders      : {total_orders:,}"
)

print(
    f"Average Order     : ₹{average_order_value:,.2f}"
)

print(
    f"Profit Margin     : {profit_margin:.2f}%"
)

# ==========================================================
# EXPORT CLEAN DATASET
# ==========================================================

output_file = (
    "sample_data/"
    "powerbi_sales_clean.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)

print(
    f"Clean dataset exported to:\n{output_file}"
)

# ==========================================================
# EXPORT SUMMARY TABLES
# ==========================================================

region_summary.to_csv(
    "sample_data/region_summary.csv",
    index=False
)

category_summary.to_csv(
    "sample_data/category_summary.csv",
    index=False
)

monthly_summary.to_csv(
    "sample_data/monthly_summary.csv",
    index=False
)

print("\nSummary datasets exported successfully.")

# ==========================================================
# POWER BI WORKFLOW
# ==========================================================

print("\n" + "=" * 80)
print("POWER BI WORKFLOW")
print("=" * 80)

print("""
1. Run this Python script.

2. Import powerbi_sales_clean.csv
   into Power BI.

3. Open Power Query.

4. Verify data types.

5. Create relationships if
   multiple tables are used.

6. Create DAX measures.

7. Build KPI cards.

8. Add charts.

9. Add slicers.

10. Publish the report to
    Power BI Service.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Practiced

✔ Python Data Preparation
✔ Data Type Conversion
✔ Date Feature Extraction
✔ Feature Engineering
✔ Aggregation
✔ KPI Calculation
✔ CSV Export
✔ Power BI Data Preparation Workflow
""")

print("\nLesson 07 Completed Successfully!")
