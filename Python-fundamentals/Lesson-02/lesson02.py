"""
Lesson 02 : Functions, Comprehensions & File Handling

Topics Covered
---------------
1. Functions
2. return Statement
3. *args
4. **kwargs
5. Lambda Functions
6. List Comprehensions
7. Dictionary Comprehensions
8. Generator Expressions
9. File Handling
10. Reading & Writing Text Files
11. CSV Files
"""

# ==================================================
# 1. FUNCTIONS
# ==================================================

print("\n========== FUNCTIONS ==========")


def greet(name):
    print(f"Hello, {name}!")


greet("Krish")


# ==================================================
# 2. RETURN
# ==================================================

print("\n========== RETURN ==========")


def add(a, b):
    return a + b


result = add(10, 20)
print("Sum =", result)


# ==================================================
# 3. *ARGS
# ==================================================

print("\n========== *ARGS ==========")


def total(*numbers):
    print("Numbers:", numbers)
    print("Sum:", sum(numbers))


total(10, 20, 30, 40)


# ==================================================
# 4. **KWARGS
# ==================================================

print("\n========== **KWARGS ==========")


def student_info(**details):
    for key, value in details.items():
        print(f"{key} : {value}")


student_info(name="Krish", age=20, course="Python")


# ==================================================
# 5. LAMBDA FUNCTION
# ==================================================

print("\n========== LAMBDA FUNCTION ==========")

square = lambda x: x * x

print(square(5))

numbers = [1, 2, 3, 4, 5]

squares = list(map(lambda x: x * x, numbers))

print(squares)


# ==================================================
# 6. LIST COMPREHENSION
# ==================================================

print("\n========== LIST COMPREHENSION ==========")

squares = [x * x for x in range(1, 11)]

print(squares)

even_numbers = [x for x in range(20) if x % 2 == 0]

print(even_numbers)


# ==================================================
# 7. DICTIONARY COMPREHENSION
# ==================================================

print("\n========== DICTIONARY COMPREHENSION ==========")

square_dict = {x: x * x for x in range(1, 6)}

print(square_dict)


# ==================================================
# 8. GENERATOR EXPRESSIONS
# ==================================================

print("\n========== GENERATOR EXPRESSIONS ==========")

generator = (x * x for x in range(1, 6))

for value in generator:
    print(value)


# ==================================================
# 9. FILE HANDLING
# ==================================================

print("\n========== FILE HANDLING ==========")

# Writing to a text file

with open("sample.txt", "w") as file:
    file.write("Hello from Python!\n")
    file.write("This file was created during Lesson 02.\n")

print("sample.txt created successfully.")


# ==================================================
# 10. READING TEXT FILE
# ==================================================

print("\n========== READING TEXT FILE ==========")

with open("sample.txt", "r") as file:
    content = file.read()

print(content)


# ==================================================
# APPENDING TO A FILE
# ==================================================

print("\n========== APPENDING ==========")

with open("sample.txt", "a") as file:
    file.write("Appending a new line.\n")

print("New line appended successfully.")


# ==================================================
# 11. CSV FILES
# ==================================================

print("\n========== CSV FILES ==========")

import csv

# Writing CSV

with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["Name", "Age", "Course"])
    writer.writerow(["Krish", 20, "Python"])
    writer.writerow(["Rahul", 21, "AI"])

print("students.csv created.")

# Reading CSV

print("\nReading CSV File")

with open("students.csv", "r") as file:
    reader = csv.reader(file)

    for row in reader:
        print(row)


# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)


print("Factorial of 5 =", factorial(5))

cubes = [x ** 3 for x in range(1, 6)]

print("Cube List:", cubes)

cube_dict = {x: x ** 3 for x in range(1, 6)}

print("Cube Dictionary:", cube_dict)


# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 02 Completed Successfully!")
