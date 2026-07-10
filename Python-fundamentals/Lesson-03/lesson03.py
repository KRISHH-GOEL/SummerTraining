"""
Lesson 03 : Object-Oriented Programming (OOP)

Topics Covered
---------------
1. Classes & Objects
2. Constructors (__init__)
3. Inheritance
4. Polymorphism
5. Encapsulation
6. Magic (Dunder) Methods
7. @staticmethod
8. @classmethod
9. @property
10. Exception Handling
11. Custom Exceptions
"""

# ==================================================
# 1. CLASSES & OBJECTS
# ==================================================

print("\n========== CLASSES & OBJECTS ==========")

class Student:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(f"Name : {self.name}")
        print(f"Age  : {self.age}")

student1 = Student("Krish", 20)
student1.display()


# ==================================================
# 2. INHERITANCE
# ==================================================

print("\n========== INHERITANCE ==========")

class Person:

    def introduce(self):
        print("I am a Person")


class Teacher(Person):

    def teach(self):
        print("I teach Python")


teacher = Teacher()

teacher.introduce()
teacher.teach()


# ==================================================
# 3. POLYMORPHISM
# ==================================================

print("\n========== POLYMORPHISM ==========")

class Animal:

    def sound(self):
        print("Animal makes a sound")


class Dog(Animal):

    def sound(self):
        print("Dog barks")


class Cat(Animal):

    def sound(self):
        print("Cat meows")


animals = [Dog(), Cat()]

for animal in animals:
    animal.sound()


# ==================================================
# 4. ENCAPSULATION
# ==================================================

print("\n========== ENCAPSULATION ==========")

class BankAccount:

    def __init__(self, balance):
        self.__balance = balance

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance


account = BankAccount(5000)

account.deposit(1000)

print("Balance :", account.get_balance())


# ==================================================
# 5. MAGIC (DUNDER) METHODS
# ==================================================

print("\n========== DUNDER METHODS ==========")

class Books:

    def __init__(self, books):
        self.books = books

    def __repr__(self):
        return f"Books({self.books})"

    def __len__(self):
        return len(self.books)

    def __iter__(self):
        return iter(self.books)


library = Books(["Python", "NumPy", "Pandas"])

print(library)

print("Total Books :", len(library))

for book in library:
    print(book)


# ==================================================
# 6. STATIC METHOD
# ==================================================

print("\n========== STATIC METHOD ==========")

class Calculator:

    @staticmethod
    def add(a, b):
        return a + b


print(Calculator.add(5, 10))


# ==================================================
# 7. CLASS METHOD
# ==================================================

print("\n========== CLASS METHOD ==========")

class Employee:

    company = "OpenAI"

    @classmethod
    def change_company(cls, new_company):
        cls.company = new_company


print(Employee.company)

Employee.change_company("TechCorp")

print(Employee.company)


# ==================================================
# 8. PROPERTY DECORATOR
# ==================================================

print("\n========== PROPERTY ==========")

class Circle:

    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return 3.14159 * self.radius ** 2


circle = Circle(7)

print(circle.area)


# ==================================================
# 9. EXCEPTION HANDLING
# ==================================================

print("\n========== EXCEPTION HANDLING ==========")

try:

    number = int(input("Enter a number: "))
    result = 100 / number

except ZeroDivisionError:

    print("Cannot divide by zero.")

except ValueError:

    print("Please enter a valid integer.")

else:

    print("Result :", result)

finally:

    print("Execution Finished.")


# ==================================================
# 10. CUSTOM EXCEPTION
# ==================================================

print("\n========== CUSTOM EXCEPTION ==========")

class InvalidAgeError(Exception):
    """Custom Exception"""
    pass


def check_age(age):

    if age < 18:
        raise InvalidAgeError("Age must be at least 18.")

    print("Access Granted.")


try:

    check_age(15)

except InvalidAgeError as e:

    print(e)


# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

class Rectangle:

    def __init__(self, length, width):
        self.length = length
        self.width = width

    @property
    def area(self):
        return self.length * self.width


rectangle = Rectangle(10, 5)

print("Area :", rectangle.area)


# ==================================================
# LESSON COMPLETED
# ==================================================

print("\nLesson 03 Completed Successfully!")
