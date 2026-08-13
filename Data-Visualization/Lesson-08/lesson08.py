"""
Lesson 08 : Tableau Fundamentals

Python Data Preparation Component

Topics Covered
---------------
1. Creating Data
2. Data Inspection
3. Date Conversion
4. Feature Engineering
5. Data Validation
6. Aggregation
7. KPI Calculation
8. Exporting Tableau-Ready Data

Purpose
-------
Prepare a clean dataset that can be imported
into Tableau for visualization and dashboard creation.

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import pandas as pd
import numpy as np

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs("sample_data", exist_ok=True)

# ==========================================================
# CREATE SALES DATA
# ==========================================================

print("=" * 80)
print("TABLEAU DATA PREPARATION")
print("=" * 80)

data = {

    "OrderID": [
        1001, 1002, 1003, 1004, 1005,
        1006, 1007, 1008, 1009, 1010,
        1011, 1012, 1013, 1014, 1015
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
        "2026-04-25",
        "2026-05-05",
        "2026-05-16",
        "2026-06-03"
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
        "Keyboard",
        "Monitor",
        "Laptop",
        "Mouse"
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
        "Accessories",
        "Electronics",
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
        "South",
        "East",
        "North",
        "West"
    ],

    "Quantity": [
        2, 5, 3, 1, 2,
        8, 4, 1, 3, 6,
        2, 5, 2, 1, 7
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
        12500,
        35000,
        140000,
        7000
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
        3750,
        7000,
        21000,
        2100
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

print("\nShape:")

print(df.shape)

print("\nData Types:")

print(df.dtypes)

print("\nMissing Values:")

print(df.isnull().sum())

print("\nDuplicate Rows:")

print(df.duplicated().sum())

# ==========================================================
# DATE CONVERSION
# ==========================================================

print("\n" + "=" * 80)
print("DATE PROCESSING")
print("=" * 80)

df["Date"] = pd.to_datetime(
    df["Date"]
)

# ==========================================================
# DATE FEATURES
# ==========================================================

df["Year"] = (
    df["Date"].dt.year
)

df["Month_Number"] = (
    df["Date"].dt.month
)

df["Month"] = (
    df["Date"].dt.month_name()
)

df["Quarter"] = (
    "Q"
    + df["Date"].dt.quarter.astype(str)
)

df["Day"] = (
    df["Date"].dt.day
)

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
# AVERAGE ORDER VALUE
# ==========================================================

df["Average_Order_Value"] = (

    df["Sales"]
    / df["Quantity"]

)

# ==========================================================
# PROFIT CATEGORY
# ==========================================================

df["Profit_Category"] = np.select(

    [

        df["Profit"] >= 10000,

        df["Profit"] >= 5000

    ],

    [

        "High Profit",

        "Medium Profit"

    ],

    default="Low Profit"

)

print("\nFeature Engineering Completed")

print(
    df[
        [
            "Date",
            "Month",
            "Quarter",
            "Profit_Margin",
            "Average_Order_Value",
            "Profit_Category"
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
# DATA QUALITY VALIDATION
# ==========================================================

print("\n" + "=" * 80)
print("DATA QUALITY VALIDATION")
print("=" * 80)

assert df["OrderID"].notna().all()

assert df["Sales"].notna().all()

assert df["Profit"].notna().all()

assert (df["Sales"] >= 0).all()

assert (df["Quantity"] > 0).all()

print("Data validation passed successfully.")

# ==========================================================
# REGION-WISE SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("REGION-WISE PERFORMANCE")
print("=" * 80)

region_summary = (

    df.groupby("Region")

    .agg(

        Total_Sales=(
            "Sales",
            "sum"
        ),

        Total_Profit=(
            "Profit",
            "sum"
        ),

        Total_Quantity=(
            "Quantity",
            "sum"
        ),

        Total_Orders=(
            "OrderID",
            "count"
        )

    )

    .reset_index()

)

region_summary["Profit_Margin"] = (

    region_summary["Total_Profit"]
    / region_summary["Total_Sales"]
    * 100

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

        Total_Sales=(
            "Sales",
            "sum"
        ),

        Total_Profit=(
            "Profit",
            "sum"
        ),

        Total_Quantity=(
            "Quantity",
            "sum"
        )

    )

    .reset_index()

)

print(category_summary)

# ==========================================================
# PRODUCT SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("PRODUCT PERFORMANCE")
print("=" * 80)

product_summary = (

    df.groupby(
        "Product"
    )

    .agg(

        Total_Sales=(
            "Sales",
            "sum"
        ),

        Total_Profit=(
            "Profit",
            "sum"
        ),

        Total_Quantity=(
            "Quantity",
            "sum"
        ),

        Orders=(
            "OrderID",
            "count"
        )

    )

    .reset_index()

)

print(product_summary)

# ==========================================================
# MONTHLY SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("MONTHLY PERFORMANCE")
print("=" * 80)

monthly_summary = (

    df.groupby(
        [
            "Year",
            "Month_Number",
            "Month",
            "Quarter"
        ]
    )

    .agg(

        Total_Sales=(
            "Sales",
            "sum"
        ),

        Total_Profit=(
            "Profit",
            "sum"
        ),

        Total_Quantity=(
            "Quantity",
            "sum"
        )

    )

    .reset_index()

    .sort_values(
        [
            "Year",
            "Month_Number"
        ]
    )

)

print(monthly_summary)

# ==========================================================
# KPI CALCULATIONS
# ==========================================================

print("\n" + "=" * 80)
print("BUSINESS KPIs")
print("=" * 80)

total_sales = df["Sales"].sum()

total_profit = df["Profit"].sum()

total_quantity = df["Quantity"].sum()

total_orders = df["OrderID"].nunique()

average_order_value = (

    total_sales
    / total_orders

)

overall_profit_margin = (

    total_profit
    / total_sales
    * 100

)

print(
    f"Total Sales          : ₹{total_sales:,.2f}"
)

print(
    f"Total Profit         : ₹{total_profit:,.2f}"
)

print(
    f"Total Quantity       : {total_quantity:,}"
)

print(
    f"Total Orders         : {total_orders:,}"
)

print(
    f"Average Order Value : ₹{average_order_value:,.2f}"
)

print(
    f"Profit Margin        : {overall_profit_margin:.2f}%"
)

# ==========================================================
# TOP PERFORMING PRODUCT
# ==========================================================

top_product = (

    product_summary
    .sort_values(
        "Total_Sales",
        ascending=False
    )
    .iloc[0]

)

print("\nTop Product by Sales:")

print(
    top_product["Product"]
)

# ==========================================================
# TOP PERFORMING REGION
# ==========================================================

top_region = (

    region_summary
    .sort_values(
        "Total_Profit",
        ascending=False
    )
    .iloc[0]

)

print("\nTop Region by Profit:")

print(
    top_region["Region"]
)

# ==========================================================
# EXPORT TABLEAU DATASET
# ==========================================================

main_file = (
    "sample_data/"
    "tableau_sales_clean.csv"
)

df.to_csv(
    main_file,
    index=False
)

print(
    f"\nMain Tableau dataset saved to:\n{main_file}"
)

# ==========================================================
# EXPORT SUMMARY DATASETS
# ==========================================================

region_summary.to_csv(

    "sample_data/"
    "region_summary.csv",

    index=False

)

category_summary.to_csv(

    "sample_data/"
    "category_summary.csv",

    index=False

)

product_summary.to_csv(

    "sample_data/"
    "product_summary.csv",

    index=False

)

monthly_summary.to_csv(

    "sample_data/"
    "monthly_summary.csv",

    index=False

)

# ==========================================================
# TABLEAU WORKFLOW
# ==========================================================

print("\n" + "=" * 80)
print("TABLEAU WORKFLOW")
print("=" * 80)

print("""
Step 1
------
Run this Python script.

Step 2
------
Open Tableau Desktop.

Step 3
------
Connect to:

tableau_sales_clean.csv

Step 4
------
Verify dimensions and measures.

Step 5
------
Create calculated fields.

Step 6
------
Create individual worksheets.

Step 7
------
Combine worksheets into a dashboard.

Step 8
------
Add filters and interactive controls.

Step 9
------
Analyze business insights.

Step 10
-------
Publish the dashboard.
""")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Practiced

✔ Tableau Data Preparation
✔ Data Validation
✔ Date Feature Engineering
✔ Calculated Business Features
✔ Aggregation
✔ KPI Calculation
✔ CSV Export
✔ Tableau Workflow
✔ Business Analysis
""")

print("\nLesson 08 Completed Successfully!")
