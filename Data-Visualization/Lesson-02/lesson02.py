"""
Lesson 02 : Basic Charts using Matplotlib

Topics Covered
---------------
1. Line Plot
2. Bar Chart
3. Scatter Plot
4. Histogram
5. Pie Chart
6. Area Chart

Dataset
-------
Titanic Dataset (Seaborn)

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs("images", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 80)
print("BASIC CHARTS USING MATPLOTLIB")
print("=" * 80)

df = sns.load_dataset("titanic")

# ==========================================================
# LINE CHART
# ==========================================================
# Best For:
# Time-series or trend analysis.

monthly_survival = (
    df.groupby("embark_town")["survived"]
    .mean()
)

plt.figure(figsize=(8,5))

plt.plot(
    monthly_survival.index,
    monthly_survival.values,
    marker="o",
    linewidth=2
)

plt.title("Average Survival Rate by Embarkation Town")

plt.xlabel("Embarkation Town")

plt.ylabel("Average Survival")

plt.grid(True)

plt.savefig("images/line_plot.png", dpi=300)

plt.show()

print("""
Business Insight
----------------
Passengers from different embarkation towns
show different average survival rates.
""")

# ==========================================================
# BAR CHART
# ==========================================================
# Best For:
# Comparing categories.

survival_class = (
    df.groupby("class")["survived"]
    .mean()
)

plt.figure(figsize=(8,5))

plt.bar(
    survival_class.index.astype(str),
    survival_class.values
)

plt.title("Average Survival Rate by Passenger Class")

plt.xlabel("Passenger Class")

plt.ylabel("Average Survival")

plt.savefig("images/bar_chart.png", dpi=300)

plt.show()

print("""
Business Insight
----------------
First-class passengers had the highest
survival probability.
""")

# ==========================================================
# SCATTER PLOT
# ==========================================================
# Best For:
# Relationship between numerical variables.

plt.figure(figsize=(8,5))

plt.scatter(
    df["age"],
    df["fare"],
    alpha=0.6
)

plt.title("Age vs Fare")

plt.xlabel("Age")

plt.ylabel("Fare")

plt.savefig("images/scatter_plot.png", dpi=300)

plt.show()

print("""
Business Insight
----------------
Higher fares generally belong to adult passengers,
although considerable variation exists.
""")

# ==========================================================
# HISTOGRAM
# ==========================================================
# Best For:
# Understanding distributions.

plt.figure(figsize=(8,5))

plt.hist(
    df["fare"].dropna(),
    bins=25
)

plt.title("Distribution of Fare")

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.savefig("images/histogram.png", dpi=300)

plt.show()

print("""
Business Insight
----------------
Fare is highly right-skewed.
Most passengers paid lower fares.
""")

# ==========================================================
# PIE CHART
# ==========================================================
# Best For:
# Percentage comparison.

gender = df["sex"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    gender.values,
    labels=gender.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Gender Distribution")

plt.savefig("images/pie_chart.png", dpi=300)

plt.show()

print("""
Business Insight
----------------
Male passengers make up the majority
of the dataset.
""")

# ==========================================================
# AREA CHART
# ==========================================================
# Best For:
# Showing cumulative trends.

fare_class = (
    df.groupby("pclass")["fare"]
    .mean()
)

plt.figure(figsize=(8,5))

plt.fill_between(
    fare_class.index,
    fare_class.values,
    alpha=0.5
)

plt.plot(
    fare_class.index,
    fare_class.values,
    marker="o"
)

plt.title("Average Fare by Passenger Class")

plt.xlabel("Passenger Class")

plt.ylabel("Average Fare")

plt.savefig("images/area_chart.png", dpi=300)

plt.show()

print("""
Business Insight
----------------
Passengers in higher classes
paid considerably higher fares.
""")

# ==========================================================
# CHART SELECTION GUIDE
# ==========================================================

print("\n" + "="*80)
print("WHEN TO USE WHICH CHART?")
print("="*80)

print("""

Line Plot
---------
✓ Time
✓ Trends

Bar Chart
----------
✓ Compare Categories

Scatter Plot
-------------
✓ Relationship
✓ Correlation

Histogram
----------
✓ Distribution

Pie Chart
----------
✓ Percentage

Area Chart
-----------
✓ Cumulative Trend

""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "="*80)
print("MINI PRACTICE")
print("="*80)

print("""

1.
Create a bar chart showing passenger count
for each embarkation town.

2.
Plot a histogram of passenger age.

3.
Create a scatter plot of age vs fare
colored by gender.

4.
Visualize passenger class using a pie chart.

5.
Compare average age by passenger class
using a bar chart.

""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "="*80)
print("LESSON SUMMARY")
print("="*80)

print("""

Skills Learned

✔ Line Plot

✔ Bar Chart

✔ Scatter Plot

✔ Histogram

✔ Pie Chart

✔ Area Chart

✔ Choosing Appropriate Charts

✔ Business Interpretation

""")

print("\nLesson 02 Completed Successfully!")
