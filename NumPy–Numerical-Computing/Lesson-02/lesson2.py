"""
Lesson 02 : Linear Algebra & Statistical Functions

Topics Covered
---------------
1. Dot Product
2. Matrix Multiplication
3. Determinant
4. Eigenvalues & Eigenvectors
5. Mean
6. Median
7. Standard Deviation
8. Variance
9. Percentile
10. Correlation Coefficient
"""

import numpy as np

# ==================================================
# LINEAR ALGEBRA
# ==================================================

print("\n========== LINEAR ALGEBRA ==========")

# --------------------------------------------------
# 1. DOT PRODUCT
# --------------------------------------------------
# np.dot() computes the dot product of two vectors.
# It is widely used in Machine Learning and Deep Learning
# for similarity calculations and matrix operations.

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

dot_product = np.dot(vector1, vector2)

print("\nDot Product")
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print("Result  :", dot_product)

# --------------------------------------------------
# 2. MATRIX MULTIPLICATION
# --------------------------------------------------
# np.matmul() performs matrix multiplication.
# The @ operator can also be used.

matrix1 = np.array([
    [1, 2],
    [3, 4]
])

matrix2 = np.array([
    [5, 6],
    [7, 8]
])

result = np.matmul(matrix1, matrix2)

print("\nMatrix Multiplication")
print(result)

print("\nUsing @ Operator")
print(matrix1 @ matrix2)

# --------------------------------------------------
# 3. DETERMINANT
# --------------------------------------------------
# np.linalg.det() calculates the determinant of a square matrix.
# The determinant indicates whether a matrix is invertible.

matrix = np.array([
    [2, 1],
    [5, 3]
])

print("\nDeterminant")
print(np.linalg.det(matrix))

# --------------------------------------------------
# 4. EIGENVALUES & EIGENVECTORS
# --------------------------------------------------
# np.linalg.eig() returns both eigenvalues and eigenvectors.
# These are widely used in PCA and dimensionality reduction.

matrix = np.array([
    [4, -2],
    [1, 1]
])

eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("\nEigenvalues")
print(eigenvalues)

print("\nEigenvectors")
print(eigenvectors)

# ==================================================
# STATISTICAL FUNCTIONS
# ==================================================

print("\n========== STATISTICAL FUNCTIONS ==========")

data = np.array([10, 20, 30, 40, 50, 60])

print("\nDataset")
print(data)

# --------------------------------------------------
# MEAN
# --------------------------------------------------
# Average value of all elements.

print("\nMean")
print(np.mean(data))

# --------------------------------------------------
# MEDIAN
# --------------------------------------------------
# Middle value after sorting.

print("\nMedian")
print(np.median(data))

# --------------------------------------------------
# STANDARD DEVIATION
# --------------------------------------------------
# Measures how spread out the data is.

print("\nStandard Deviation")
print(np.std(data))

# --------------------------------------------------
# VARIANCE
# --------------------------------------------------
# Square of the standard deviation.

print("\nVariance")
print(np.var(data))

# --------------------------------------------------
# PERCENTILE
# --------------------------------------------------
# Finds the value below which a given percentage
# of observations fall.

print("\n25th Percentile")
print(np.percentile(data, 25))

print("\n50th Percentile")
print(np.percentile(data, 50))

print("\n90th Percentile")
print(np.percentile(data, 90))

# --------------------------------------------------
# CORRELATION COEFFICIENT
# --------------------------------------------------
# Measures the strength of the relationship
# between two datasets.
#
# +1 → Perfect Positive Correlation
#  0 → No Correlation
# -1 → Perfect Negative Correlation

math_marks = np.array([80, 85, 90, 95, 100])
science_marks = np.array([78, 84, 88, 94, 98])

correlation = np.corrcoef(math_marks, science_marks)

print("\nCorrelation Matrix")
print(correlation)

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

sales = np.array([120, 150, 180, 170, 210, 240])

print("\nSales Data")
print(sales)

print("\nAverage Sales")
print(np.mean(sales))

print("\nHighest Sale")
print(np.max(sales))

print("\nLowest Sale")
print(np.min(sales))

print("\nStandard Deviation")
print(np.std(sales))

print("\n75th Percentile")
print(np.percentile(sales, 75))

# Matrix Practice

matrix_a = np.array([
    [1, 0],
    [2, 3]
])

matrix_b = np.array([
    [4, 1],
    [2, 5]
])

print("\nMatrix Multiplication Practice")
print(matrix_a @ matrix_b)

print("\nDeterminant of Matrix A")
print(np.linalg.det(matrix_a))

# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 02 Completed Successfully!")
