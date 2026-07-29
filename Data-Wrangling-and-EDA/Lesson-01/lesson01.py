"""
Lesson 01 : Data Collection & Data Sources

Topics Covered
---------------
1. Structured Data
2. Semi-Structured Data
3. Unstructured Data
4. Public Dataset Sources
5. Loading Data into Pandas

Author : Krish Goel
Repository : Summer Training
"""

import pandas as pd

# ==================================================
# INTRODUCTION
# ==================================================

print("=" * 70)
print("DATA COLLECTION & DATA SOURCES")
print("=" * 70)

print("""
Data Collection is the first stage of every
Data Science and Machine Learning project.

The goal is to gather relevant, high-quality
data from reliable sources for analysis.
""")

# ==================================================
# TYPES OF DATA
# ==================================================

print("\n========== TYPES OF DATA ==========")

print("""
1. Structured Data
------------------
Organized into rows and columns.

Examples:
- Excel files
- SQL databases
- CSV files
""")

print("""
2. Semi-Structured Data
-----------------------
Contains tags or keys but no fixed table structure.

Examples:
- JSON
- XML
- HTML
""")

print("""
3. Unstructured Data
--------------------
No predefined format.

Examples:
- Images
- Videos
- Audio
- PDFs
- Emails
- Social Media Posts
""")

# ==================================================
# PUBLIC DATA SOURCES
# ==================================================

print("\n========== PUBLIC DATA SOURCES ==========")

sources = {
    "Kaggle": "https://www.kaggle.com/datasets",
    "UCI ML Repository": "https://archive.ics.uci.edu/ml",
    "data.gov.in": "https://data.gov.in",
    "World Bank": "https://data.worldbank.org"
}

for source, website in sources.items():
    print(f"{source:<20} : {website}")

# ==================================================
# LOADING DATA
# ==================================================

print("\n========== LOADING DATA ==========")

print("""
Common Pandas Functions

pd.read_csv("file.csv")

pd.read_excel("file.xlsx")

pd.read_json("file.json")
""")

# --------------------------------------------------
# EXAMPLE DATAFRAME
# --------------------------------------------------

students = pd.DataFrame({
    "Name": ["Krish", "Adi", "Varun"],
    "Age": [20, 21, 20],
    "Course": ["AI", "Python", "ML"]
})

print("\nSample Dataset")
print(students)

print("\nDataset Shape")
print(students.shape)

print("\nColumn Names")
print(students.columns.tolist())

print("\nData Types")
print(students.dtypes)

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

print("""
Exercise 1
----------
Visit Kaggle and search for:
- House Price Prediction
- Titanic
- Heart Disease
- Customer Churn

Exercise 2
----------
Identify whether each dataset is:

✓ Structured
✓ Semi-Structured
✓ Unstructured
""")

print("\nLesson 01 Completed Successfully!")
