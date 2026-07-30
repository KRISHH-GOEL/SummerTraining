"""
Lesson 02 : Web Scraping, REST APIs & SQL

Topics Covered
---------------
1. HTTP Requests
2. Web Scraping
3. BeautifulSoup
4. REST APIs
5. JSON Parsing
6. SQLAlchemy
7. pandas.read_sql()

Author : Krish Goel
Repository : Summer Training
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ==================================================
# HTTP REQUESTS
# ==================================================

print("=" * 70)
print("WEB SCRAPING, REST APIs & SQL")
print("=" * 70)

# --------------------------------------------------
# requests.get()
# --------------------------------------------------
# Sends an HTTP GET request to retrieve a web page.

url = "https://example.com"

try:

    response = requests.get(url)

    print("\n========== HTTP REQUEST ==========")

    print("Status Code :", response.status_code)

    print("Request Successful :", response.ok)

except Exception as e:

    print("Error :", e)

# ==================================================
# BEAUTIFULSOUP
# ==================================================

print("\n========== BEAUTIFULSOUP ==========")

# BeautifulSoup parses HTML documents and allows
# easy extraction of information.

soup = BeautifulSoup(response.text, "html.parser")

print("\nPage Title")

print(soup.title.text)

print("\nFirst Heading")

heading = soup.find("h1")

if heading:
    print(heading.text)

print("\nAll Links")

for link in soup.find_all("a"):
    print(link.get("href"))

# ==================================================
# REST APIs
# ==================================================

print("\n========== REST API ==========")

# Public testing API

api_url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(api_url)

users = response.json()

print("\nFirst User")

print(users[0])

print("\nUser Names")

for user in users:
    print(user["name"])

# ==================================================
# JSON TO DATAFRAME
# ==================================================

print("\n========== JSON TO DATAFRAME ==========")

df = pd.DataFrame(users)

print(df.head())

# ==================================================
# SQL DATABASES
# ==================================================

print("\n========== SQL DATABASE ==========")

print("""
Example:

from sqlalchemy import create_engine

engine = create_engine(
    "sqlite:///students.db"
)

query = "SELECT * FROM students"

df = pd.read_sql(query, engine)

print(df.head())
""")

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

print("""
Exercise 1
----------
Visit:
https://jsonplaceholder.typicode.com

Explore different API endpoints.

Exercise 2
----------
Try extracting:

- Names
- Emails
- Company Names

using response.json().

Exercise 3
----------
Visit:

https://books.toscrape.com/

Extract:

- Book Titles
- Prices
- Ratings

using BeautifulSoup.
""")

print("\nLesson 02 Completed Successfully!")
