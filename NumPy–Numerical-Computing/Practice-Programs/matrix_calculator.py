"""
Project : Matrix Calculator
Module  : NumPy

Description:
------------
This project demonstrates common linear algebra
operations using NumPy.

Concepts Used:
--------------
- Dot Product
- Matrix Multiplication
- Determinant
- Eigenvalues
- Eigenvectors
"""

import numpy as np

print("=" * 60)
print("MATRIX CALCULATOR")
print("=" * 60)

# --------------------------------------------------
# MATRICES
# --------------------------------------------------

matrix_a = np.array([
    [2, 3],
    [4, 5]
])

matrix_b = np.array([
    [1, 2],
    [3, 4]
])

print("\nMatrix A")

print(matrix_a)

print("\nMatrix B")

print(matrix_b)

# --------------------------------------------------
# MATRIX MULTIPLICATION
# --------------------------------------------------

print("\nMatrix Multiplication")

print(matrix_a @ matrix_b)

# --------------------------------------------------
# DOT PRODUCT
# --------------------------------------------------

vector1 = np.array([1, 2, 3])

vector2 = np.array([4, 5, 6])

print("\nDot Product")

print(np.dot(vector1, vector2))

# --------------------------------------------------
# DETERMINANT
# --------------------------------------------------

print("\nDeterminant of Matrix A")

print(np.linalg.det(matrix_a))

# --------------------------------------------------
# EIGENVALUES & EIGENVECTORS
# --------------------------------------------------

eigenvalues, eigenvectors = np.linalg.eig(matrix_a)

print("\nEigenvalues")

print(eigenvalues)

print("\nEigenvectors")

print(eigenvectors)

# --------------------------------------------------
# MATRIX ADDITION
# --------------------------------------------------

print("\nMatrix Addition")

print(matrix_a + matrix_b)

# --------------------------------------------------
# MATRIX SUBTRACTION
# --------------------------------------------------

print("\nMatrix Subtraction")

print(matrix_a - matrix_b)

print("\nProgram Completed Successfully!")
