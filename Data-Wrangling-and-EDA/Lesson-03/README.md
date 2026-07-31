# Lesson 03 – Data Cleaning & Data Preprocessing

## 📌 Objective

The objective of this lesson was to understand the importance of data cleaning and preprocessing in the Data Science workflow.

Real-world datasets are often incomplete, inconsistent, and noisy. Before performing analysis or building machine learning models, data must be cleaned to improve quality and reliability.

This lesson focuses on identifying missing values, handling duplicates, correcting structural inconsistencies, parsing date-time data, and validating dataset schemas.

---

## 📚 Topics Covered

### Missing Values

- What are Missing Values?
- Types of Missing Data
  - MCAR (Missing Completely at Random)
  - MAR (Missing at Random)
  - MNAR (Missing Not at Random)

### Missing Value Handling

- Detect Missing Values
- Remove Missing Values
- Fill Missing Values
- Mean Imputation
- Median Imputation
- Mode Imputation
- KNN Imputer
- Iterative Imputer

### Structural Errors

- Case Normalization
- Removing Extra Whitespaces
- Correcting Typographical Errors
- Fixing Unit Inconsistencies

### Duplicate Records

- Detecting Duplicate Rows
- Removing Duplicate Rows

### Date & Time Processing

- Datetime Parsing
- Extracting Year
- Extracting Month
- Extracting Day
- Extracting Day of Week
- Extracting Quarter

### Schema Validation

- Checking Data Types
- Enforcing Correct Data Types

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Identify different types of missing values.
- Apply multiple missing-value imputation strategies.
- Detect and remove duplicate records.
- Standardize inconsistent text data.
- Parse and manipulate datetime columns.
- Extract useful date-related features.
- Validate and correct dataset data types.

---

## 📝 Dataset Used

Titanic Dataset (Seaborn)

---

## ⚡ Quick Revision

| Function | Purpose |
|----------|---------|
| `isnull()` | Detect missing values |
| `dropna()` | Remove missing values |
| `fillna()` | Replace missing values |
| `SimpleImputer` | Mean / Median / Mode imputation |
| `KNNImputer` | KNN-based imputation |
| `IterativeImputer` | Predictive imputation |
| `drop_duplicates()` | Remove duplicate rows |
| `str.lower()` | Convert text to lowercase |
| `str.strip()` | Remove whitespaces |
| `pd.to_datetime()` | Convert to datetime |
| `.dt.year` | Extract year |
| `.astype()` | Change data type |

---

## 📦 Libraries Used

- pandas
- numpy
- seaborn
- scikit-learn

---

## 🚀 Next Lesson

In the next lesson, I will learn various techniques for detecting and treating outliers, transforming numerical data, and preparing datasets for feature engineering.
