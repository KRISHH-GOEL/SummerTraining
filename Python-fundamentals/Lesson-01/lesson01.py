"""
Lesson 01 : Python Fundamentals

Topics Covered:
1. Variables
2. Data Types
3. Type Casting
4. Input & Output
5. Operators
6. Conditional Statements
7. Loops
8. break
9. continue
10. pass
"""

# ==================================================
# 1. VARIABLES
# ==================================================

name = "Krish"
age = 20
cgpa = 8.7
is_student = True

print("Variables")
print(name, age, cgpa, is_student)

# ==================================================
# 2. DATA TYPES
# ==================================================

print("\nData Types")
print(type(name))
print(type(age))
print(type(cgpa))
print(type(is_student))

# ==================================================
# 3. TYPE CASTING
# ==================================================

num = "100"

integer_num = int(num)
float_num = float(num)

print("\nType Casting")
print(integer_num)
print(float_num)

# ==================================================
# 4. INPUT & OUTPUT
# ==================================================

print("\nInput & Output")

city = input("Enter your city: ")

print("Hello! You are from", city)

# ==================================================
# 5. OPERATORS
# ==================================================

a = 15
b = 4

print("\nArithmetic Operators")
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)
print(a ** b)

print("\nComparison Operators")
print(a > b)
print(a == b)
print(a != b)

print("\nLogical Operators")
print(a > 10 and b < 5)
print(a < 10 or b < 5)
print(not (a > b))

# ==================================================
# 6. CONDITIONAL STATEMENTS
# ==================================================

marks = int(input("\nEnter your marks: "))

print("\nConditional Statements")

if marks >= 90:
    print("Grade A")
elif marks >= 75:
    print("Grade B")
elif marks >= 50:
    print("Grade C")
else:
    print("Fail")

# ==================================================
# 7. FOR LOOP
# ==================================================

print("\nFor Loop")

for i in range(1, 6):
    print(i)

# ==================================================
# 8. WHILE LOOP
# ==================================================

print("\nWhile Loop")

count = 1

while count <= 5:
    print(count)
    count += 1

# ==================================================
# 9. BREAK
# ==================================================

print("\nBreak Example")

for i in range(1, 11):
    if i == 6:
        break
    print(i)

# ==================================================
# 10. CONTINUE
# ==================================================

print("\nContinue Example")

for i in range(1, 11):
    if i == 6:
        continue
    print(i)

# ==================================================
# 11. PASS
# ==================================================

print("\nPass Example")

for i in range(1, 6):
    if i == 3:
        pass
    print(i)

print("\nLesson 01 Completed Successfully!")
