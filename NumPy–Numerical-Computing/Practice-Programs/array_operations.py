"""
Project : Array Operations
Module  : NumPy

Description:
------------
This program demonstrates common NumPy array operations
such as creation, indexing, slicing, boolean indexing,
fancy indexing, broadcasting, and vectorized operations.

Concepts Used:
--------------
- np.array()
- np.arange()
- np.linspace()
- Indexing
- Slicing
- Boolean Indexing
- Fancy Indexing
- Broadcasting
- Vectorized Operations
"""

import numpy as np

print("=" * 60)
print("ARRAY OPERATIONS USING NUMPY")
print("=" * 60)

# --------------------------------------------------
# ARRAY CREATION
# --------------------------------------------------

print("\nCreating Arrays")

arr1 = np.array([10, 20, 30, 40, 50])

arr2 = np.arange(1, 11)

arr3 = np.linspace(0, 100, 6)

print("Array 1 :", arr1)
print("Array 2 :", arr2)
print("Array 3 :", arr3)

# --------------------------------------------------
# INDEXING
# --------------------------------------------------

print("\nIndexing")

print("First Element :", arr1[0])
print("Last Element :", arr1[-1])

# --------------------------------------------------
# SLICING
# --------------------------------------------------

print("\nSlicing")

print(arr1[1:4])

# --------------------------------------------------
# BOOLEAN INDEXING
# --------------------------------------------------

print("\nNumbers Greater Than 25")

print(arr1[arr1 > 25])

# --------------------------------------------------
# FANCY INDEXING
# --------------------------------------------------

print("\nFancy Indexing")

print(arr1[[0, 2, 4]])

# --------------------------------------------------
# BROADCASTING
# --------------------------------------------------

print("\nBroadcasting")

print(arr1 + 10)

# --------------------------------------------------
# VECTORIZED OPERATIONS
# --------------------------------------------------

print("\nVectorized Operations")

print("Square :", arr2 ** 2)
print("Cube :", arr2 ** 3)

print("\nProgram Completed Successfully!")
