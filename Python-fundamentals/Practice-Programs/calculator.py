"""
Program : Calculator
Module  : Python Fundamentals (Module 01)

Description:
------------
A simple command-line calculator that performs basic arithmetic
operations based on the user's choice.

Concepts Used:
--------------
- Variables
- Input & Output
- Operators
- Conditional Statements
- Functions
- Exception Handling
"""

# ==================================================
# FUNCTIONS
# ==================================================

def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        return "Error! Division by zero is not allowed."
    return a / b


# ==================================================
# MAIN PROGRAM
# ==================================================

print("=" * 40)
print("        SIMPLE CALCULATOR")
print("=" * 40)

print("\nSelect an Operation:")
print("1. Addition")
print("2. Subtraction")
print("3. Multiplication")
print("4. Division")

try:
    choice = int(input("\nEnter your choice (1-4): "))

    if choice not in [1, 2, 3, 4]:
        print("Invalid Choice!")
        exit()

    num1 = float(input("Enter First Number : "))
    num2 = float(input("Enter Second Number: "))

    if choice == 1:
        print(f"\nResult = {add(num1, num2)}")

    elif choice == 2:
        print(f"\nResult = {subtract(num1, num2)}")

    elif choice == 3:
        print(f"\nResult = {multiply(num1, num2)}")

    elif choice == 4:
        print(f"\nResult = {divide(num1, num2)}")

except ValueError:
    print("Please enter valid numeric input.")

print("\nProgram Executed Successfully.")
