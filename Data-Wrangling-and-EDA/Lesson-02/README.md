# Lesson 02 – Web Scraping, REST APIs & SQL Data Collection

## 📌 Objective

The objective of this lesson was to learn how to collect data from online sources such as websites, REST APIs, and SQL databases. These techniques are essential for acquiring real-world data when datasets are not readily available as CSV or Excel files.

---

## 📚 Topics Covered

### Web Scraping

- What is Web Scraping?
- HTTP Requests
- The `requests` library
- Parsing HTML using BeautifulSoup
- Finding HTML Elements
- Extracting Text and Links

### REST APIs

- What is an API?
- HTTP Methods (GET, POST, PUT, DELETE)
- Sending GET Requests
- Reading JSON Responses
- JSON Parsing

### SQL Databases

- Introduction to SQLAlchemy
- Connecting to a SQL Database
- Reading SQL Tables using `pandas.read_sql()`

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Fetch web pages using the `requests` library.
- Parse HTML documents using BeautifulSoup.
- Extract useful information from web pages.
- Retrieve data from REST APIs.
- Parse JSON responses into Python objects.
- Connect to SQL databases.
- Load SQL tables directly into Pandas DataFrames.

---

## 📝 Practice

The accompanying `lesson02_web_scraping_api_sql.py` file contains examples demonstrating every concept covered in this lesson.

---

## ⚡ Quick Revision

| Function | Purpose |
|----------|---------|
| `requests.get()` | Send HTTP GET request |
| `response.status_code` | HTTP response code |
| `response.text` | HTML content |
| `response.json()` | Convert JSON response to Python dictionary |
| `BeautifulSoup()` | Parse HTML |
| `find()` | Find first HTML element |
| `find_all()` | Find multiple HTML elements |
| `pd.read_sql()` | Read SQL table into DataFrame |

---

## 📦 Libraries Used

- requests
- beautifulsoup4
- pandas
- sqlalchemy

Install them using:

```bash
pip install requests beautifulsoup4 sqlalchemy
```

---

## 🚀 Next Lesson

In the next lesson, I will learn advanced data cleaning techniques including handling missing values, duplicate records, structural inconsistencies, and imputation methods.
