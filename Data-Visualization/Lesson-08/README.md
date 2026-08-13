# Lesson 08 – Tableau Fundamentals

## 📌 Objective

The objective of this lesson was to understand the fundamentals of Tableau and learn how it can be used to connect to data, analyze information, create visualizations, and build interactive dashboards.

Tableau is a Business Intelligence and Data Visualization platform widely used for exploratory analysis, reporting, and business decision-making.

---

## 📚 Topics Covered

### Introduction to Tableau

- What is Tableau?
- Tableau Desktop
- Tableau Public
- Tableau Server / Tableau Cloud
- Tableau workflow

### Connecting to Data

- Excel
- CSV
- SQL databases
- Other data sources
- Python-prepared datasets

### Tableau Data Model

- Tables
- Relationships
- Joins
- Unions
- Data model

### Dimensions and Measures

- Dimensions
- Measures
- Discrete fields
- Continuous fields
- Dimensions vs Measures

### Basic Visualizations

- Bar Chart
- Line Chart
- Scatter Plot
- Pie Chart
- Maps
- Tables
- Highlight Tables

### Calculated Fields

- Creating calculated fields
- Arithmetic calculations
- Conditional calculations
- Aggregations

### Filters

- Dimension filters
- Measure filters
- Context filters
- Interactive filters

### Dashboards

- Combining worksheets
- Dashboard layout
- Dashboard filters
- Interactive dashboards
- Business storytelling

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Understand the Tableau interface.
- Connect Tableau to different data sources.
- Understand the difference between dimensions and measures.
- Understand discrete and continuous fields.
- Create basic visualizations.
- Create calculated fields.
- Apply filters.
- Combine worksheets into dashboards.
- Design interactive analytical dashboards.

---

## 🧠 Tableau Data Concepts

### Dimension

Dimensions are generally categorical fields used to describe or segment data.

Examples:

```text
Product
Region
Category
Customer
Gender
Country
```

### Measure

Measures are generally numerical fields that can be aggregated.

Examples:

```text
Sales
Profit
Quantity
Revenue
Cost
```

---

## 🔵 Discrete vs Continuous

### Discrete

Discrete fields create separate categories.

Example:

```text
North
South
East
West
```

### Continuous

Continuous fields represent a continuous range of values.

Example:

```text
Sales
Profit
Age
Date
```

---

## 🧮 Example Calculated Fields

### Profit Margin

```text
SUM([Profit]) / SUM([Sales])
```

### Average Order Value

```text
SUM([Sales]) / COUNTD([OrderID])
```

### Profit Category

```text
IF [Profit] > 10000 THEN
    "High Profit"
ELSEIF [Profit] > 5000 THEN
    "Medium Profit"
ELSE
    "Low Profit"
END
```

---

## 📊 Dashboard Project

Using the provided sales dataset, create a Tableau dashboard containing:

### KPI Cards

- Total Sales
- Total Profit
- Total Orders
- Profit Margin

### Charts

- Monthly Sales Trend
- Sales by Category
- Profit by Region
- Product Performance
- Sales vs Profit Scatter Plot

### Filters

- Region
- Category
- Product
- Date

---

## 🔄 Tableau Workflow

```text
Connect to Data
      ↓
Data Preparation
      ↓
Data Model
      ↓
Dimensions & Measures
      ↓
Calculated Fields
      ↓
Worksheets
      ↓
Filters
      ↓
Dashboard
      ↓
Insights
      ↓
Publish
```

---

## 📦 Dataset

The accompanying Python script generates:

```text
sample_data/
│
├── tableau_sales_clean.csv
├── region_summary.csv
├── category_summary.csv
└── monthly_summary.csv
```

The primary dataset for Tableau is:

```text
tableau_sales_clean.csv
```

---

## 🛠️ Python Preparation

Run:

```bash
python tableau_data_preparation.py
```

The script:

- Creates sample sales data.
- Converts dates.
- Creates date features.
- Creates profit margin.
- Creates average order value.
- Performs data validation.
- Generates summary datasets.
- Exports Tableau-ready CSV files.

---

## 📝 Practical Task

Build a complete Tableau Sales Dashboard.

### Step 1 – Connect Data

Import:

```text
tableau_sales_clean.csv
```

### Step 2 – Verify Fields

Identify:

```text
Dimensions:
- Product
- Category
- Region
- Month Name
- Quarter

Measures:
- Sales
- Profit
- Quantity
- Profit Margin
```

### Step 3 – Create Calculated Fields

Create:

```text
Profit Margin
Average Order Value
Profit Category
```

### Step 4 – Create Worksheets

Build:

```text
1. Sales KPI
2. Profit KPI
3. Orders KPI
4. Monthly Sales
5. Category Sales
6. Regional Profit
7. Product Performance
8. Sales vs Profit
```

### Step 5 – Create Dashboard

Combine the worksheets into one interactive dashboard.

Add:

```text
Region Filter
Category Filter
Product Filter
Date Filter
```

### Step 6 – Business Insights

Answer:

- Which category generates the highest sales?
- Which region generates the highest profit?
- Which products perform best?
- How do sales change over time?
- Which products have high sales but low profit?
- Which regions require attention?

---

## ⭐ Key Takeaway

Tableau is not just about creating charts.

The complete workflow is:

```text
Data
 ↓
Analysis
 ↓
Visualization
 ↓
Dashboard
 ↓
Business Insight
 ↓
Decision
```

The goal is to convert raw data into information that can support business decisions.

---

## 🚀 Module 4 Completion

After completing this lesson, the Data Visualization module covers:

- Matplotlib
- Seaborn
- Plotly
- Dash
- Power BI
- Tableau

This provides exposure to both Python-based visualization and industry BI platforms.
