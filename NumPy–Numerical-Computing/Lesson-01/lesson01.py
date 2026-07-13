"""
Lesson 01 : NumPy Basics & Numerical Computing

Topics Covered
---------------
1. Introduction to NumPy
2. Creating Arrays
3. Array Indexing
4. Array Slicing
5. Boolean Indexing
6. Fancy Indexing
7. Broadcasting
8. Vectorized Operations
"""

import numpy as np

# ==================================================
# 1. INTRODUCTION TO NUMPY
# ==================================================

print("\n========== INTRODUCTION ==========")

python_list = [1, 2, 3, 4, 5]

numpy_array = np.array(python_list)

print("Python List :", python_list)
print("NumPy Array :", numpy_array)

# ==================================================
# 2. ARRAY CREATION
# ==================================================

print("\n========== ARRAY CREATION ==========")

arr1 = np.array([10, 20, 30, 40])

zeros = np.zeros((2, 3))

ones = np.ones((3, 2))

arange = np.arange(1, 11)

linspace = np.linspace(0, 1, 5)

random_array = np.random.rand(3, 3)

print("\nnp.array()")
print(arr1)

print("\nnp.zeros()")
print(zeros)

print("\nnp.ones()")
print(ones)

print("\nnp.arange()")
print(arange)

print("\nnp.linspace()")
print(linspace)

print("\nnp.random.rand()")
print(random_array)

# ==================================================
# 3. ARRAY INDEXING
# ==================================================

print("\n========== ARRAY INDEXING ==========")

arr = np.array([10, 20, 30, 40, 50])

print("First Element :", arr[0])
print("Last Element  :", arr[-1])

# ==================================================
# 4. ARRAY SLICING
# ==================================================

print("\n========== ARRAY SLICING ==========")

print(arr[1:4])

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("\nOriginal Matrix")
print(matrix)

print("\nFirst Row")
print(matrix[0])

print("\nSecond Column")
print(matrix[:, 1])

print("\nSub Matrix")
print(matrix[0:2, 1:3])

# ==================================================
# 5. BOOLEAN INDEXING
# ==================================================

print("\n========== BOOLEAN INDEXING ==========")

numbers = np.array([5, 10, 15, 20, 25])

print(numbers[numbers > 15])

# ==================================================
# 6. FANCY INDEXING
# ==================================================

print("\n========== FANCY INDEXING ==========")

arr = np.array([100, 200, 300, 400, 500])

print(arr[[0, 2, 4]])

# ==================================================
# 7. BROADCASTING
# ==================================================

print("\n========== BROADCASTING ==========")

array = np.array([1, 2, 3, 4])

print("Original :", array)

print("Add Scalar")
print(array + 10)

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

vector = np.array([10, 20, 30])

print("\nMatrix")
print(matrix)

print("\nVector")
print(vector)

print("\nBroadcast Result")
print(matrix + vector)

# ==================================================
# 8. VECTORIZED OPERATIONS
# ==================================================

print("\n========== VECTORIZED OPERATIONS ==========")

a = np.array([1, 2, 3])

b = np.array([4, 5, 6])

print("Addition")
print(a + b)

print("\nSubtraction")
print(a - b)

print("\nMultiplication")
print(a * b)

print("\nDivision")
print(a / b)

print("\nPower")
print(a ** 2)

print("\nSquare Root")
print(np.sqrt(a))

print("\nExponential")
print(np.exp(a))

print("\nComparison")
print(a > 2)

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

marks = np.array([45, 88, 72, 91, 63, 54])

print("Marks :", marks)

print("\nStudents Scoring Above 70")
print(marks[marks > 70])

numbers = np.arange(1, 11)

print("\nNumbers")
print(numbers)

print("\nSquares")
print(numbers ** 2)

print("\nCube")
print(numbers ** 3)

# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 01 Completed Successfully!")
