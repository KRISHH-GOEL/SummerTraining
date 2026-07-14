"""
Project : Student Marks Analysis
Module  : NumPy

Description:
------------
This project analyzes student marks using NumPy's
statistical functions.

Concepts Used:
--------------
- Mean
- Median
- Standard Deviation
- Variance
- Percentiles
- Boolean Indexing
- Vectorized Operations
"""

import numpy as np

print("=" * 60)
print("STUDENT MARKS ANALYSIS")
print("=" * 60)

marks = np.array([78, 65, 92, 88, 55, 73, 95, 60, 81, 69])

print("\nStudent Marks")

print(marks)

# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

print("\nAverage Marks")

print(np.mean(marks))

print("\nMedian")

print(np.median(marks))

print("\nHighest Marks")

print(np.max(marks))

print("\nLowest Marks")

print(np.min(marks))

print("\nStandard Deviation")

print(np.std(marks))

print("\nVariance")

print(np.var(marks))

# --------------------------------------------------
# PERCENTILES
# --------------------------------------------------

print("\n25th Percentile")

print(np.percentile(marks, 25))

print("\n50th Percentile")

print(np.percentile(marks, 50))

print("\n90th Percentile")

print(np.percentile(marks, 90))

# --------------------------------------------------
# BOOLEAN INDEXING
# --------------------------------------------------

print("\nStudents Scoring Above 80")

print(marks[marks > 80])

print("\nStudents Scoring Below 60")

print(marks[marks < 60])

# --------------------------------------------------
# VECTORIZED OPERATIONS
# --------------------------------------------------

print("\nGrace Marks (+5)")

print(marks + 5)

print("\nProgram Completed Successfully!")
