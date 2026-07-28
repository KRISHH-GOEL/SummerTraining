"""
Project : Employee Data Management
"""

import pandas as pd

employee = pd.DataFrame({

    "ID":[101,102,103,104],

    "Name":["Krish","Adi","Varun","Rahul"],

    "Department":["IT","HR","Finance","IT"],

    "Salary":[65000,45000,72000,60000]

})

department = pd.DataFrame({

    "ID":[101,102,103,104],

    "City":["Delhi","Mumbai","Pune","Jaipur"]

})

print("="*60)
print("EMPLOYEE MANAGEMENT")
print("="*60)

# Merge

merged = pd.merge(employee,department,on="ID")

print("\nMerged Data")

print(merged)

# Apply

merged["Bonus"] = merged["Salary"].apply(lambda x: x*0.10)

# Map

category = {

"IT":"Technical",

"HR":"Management",

"Finance":"Accounts"

}

merged["Category"] = merged["Department"].map(category)

print("\nUpdated Dataset")

print(merged)

# --------------------------------------------------
# Time Series
# --------------------------------------------------

dates = pd.date_range("2026-01-01",periods=4)

attendance = pd.DataFrame({

"Attendance":[1,1,0,1]

},index=dates)

print("\nAttendance")

print(attendance)

print("\nShift")

attendance["Previous Day"] = attendance["Attendance"].shift()

print(attendance)

print("\nRolling Average")

print(attendance.rolling(2).mean())
