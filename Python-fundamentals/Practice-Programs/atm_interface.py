"""
Project : ATM Interface
Module  : Python Fundamentals (Module 01)

Description:
------------
A simple ATM Interface that simulates basic banking operations.

Features:
---------
- Card Validation
- PIN Verification
- Cash Withdrawal
- Balance Inquiry
- Cash Deposit

Concepts Used:
--------------
- Variables
- Tuples
- Loops
- Conditional Statements
- User Input
- Boolean Variables
"""

print("=" * 50)
print("           WELCOME TO THE ATM")
print("=" * 50)

print("\nPlease insert your card...")
print("Card inserted successfully.\nProcessing your request...\n")

# ==================================================
# CUSTOMER DATABASE
# (Card Number, Name, PIN, Balance)
# ==================================================

customers = (
    (123, "Krish", 56, 5000),
    (456, "Adi", 66, 4000),
    (789, "Varun", 86, 3000),
)

# ==================================================
# LANGUAGE SELECTION
# ==================================================

language = input(
    "Choose Language:\n"
    "1. Hindi\n"
    "2. English\n\n"
    "Enter Language: "
)

print(f"\nSelected Language: {language}")

# ==================================================
# CARD VALIDATION
# ==================================================

card_number = int(input("\nEnter Card Number: "))

customer_found = False

for customer in customers:

    if card_number == customer[0]:

        customer_found = True

        print(f"\nWelcome, {customer[1]}!")

        # PIN Verification
        pin = int(input("Enter PIN: "))

        if pin != customer[2]:
            print("Incorrect PIN!")
            break

        # Menu
        print("\nSelect an Option")
        print("1. Withdraw Cash")
        print("2. Check Balance")
        print("3. Deposit Cash")

        choice = input("\nEnter Your Choice: ")

        # Withdrawal
        if choice == "1":

            amount = int(input("Enter Amount: "))

            if amount <= customer[3]:

                remaining_balance = customer[3] - amount

                print("\nPlease collect your cash.")
                print(f"Remaining Balance: ₹{remaining_balance}")

            else:

                print("\nInsufficient Balance!")

        # Balance Inquiry
        elif choice == "2":

            print(f"\nAvailable Balance: ₹{customer[3]}")

        # Deposit
        elif choice == "3":

            deposit = int(input("Enter Deposit Amount: "))

            updated_balance = customer[3] + deposit

            print(f"\nAmount Deposited Successfully.")
            print(f"Updated Balance: ₹{updated_balance}")

        else:

            print("\nInvalid Choice!")

        break

# ==================================================
# INVALID CUSTOMER
# ==================================================

if not customer_found:
    print("\nInvalid Card Number!")
