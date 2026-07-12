"""
Project : Rectangle Calculator
Module  : Python Fundamentals (Module 01)

Description:
------------
A simple Object-Oriented Python program that calculates the
area, perimeter, and diagonal of a rectangle.

Concepts Used:
--------------
- Classes & Objects
- Constructors (__init__)
- Instance Methods
- Math Module
- User Input
"""

import math


class Rectangle:
    """A class representing a rectangle."""

    def __init__(self, length, breadth):
        self.length = length
        self.breadth = breadth

    def draw(self):
        print(f"\nDrawing a rectangle with:")
        print(f"Length  : {self.length}")
        print(f"Breadth : {self.breadth}")

    def area(self):
        return self.length * self.breadth

    def perimeter(self):
        return 2 * (self.length + self.breadth)

    def diagonal(self):
        return math.sqrt(self.length ** 2 + self.breadth ** 2)


# ==================================================
# MAIN PROGRAM
# ==================================================

print("=" * 50)
print("         RECTANGLE CALCULATOR")
print("=" * 50)

length = float(input("\nEnter Length  : "))
breadth = float(input("Enter Breadth : "))

rectangle = Rectangle(length, breadth)

rectangle.draw()

print(f"\nArea      : {rectangle.area():.2f}")
print(f"Perimeter : {rectangle.perimeter():.2f}")
print(f"Diagonal  : {rectangle.diagonal():.2f}")

print("\nProgram Executed Successfully.")
