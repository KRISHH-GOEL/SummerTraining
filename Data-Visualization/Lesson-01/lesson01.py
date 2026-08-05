"""
Lesson 01 : Matplotlib Fundamentals

Topics Covered
---------------
1. Figure
2. Axes
3. Artist
4. Basic Line Plot
5. Labels
6. Title
7. Legend
8. Grid
9. Figure Size
10. DPI
11. Saving Figures

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Create folder for storing plots
# ----------------------------------------------------------

os.makedirs("images", exist_ok=True)

# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("MATPLOTLIB FUNDAMENTALS")
print("=" * 80)

print("""
Matplotlib is Python's most popular visualization library.

It is used for:

• Data Analysis
• Machine Learning
• Scientific Computing
• Business Reporting
• Dashboards

Everything drawn in Matplotlib is called an Artist.

The hierarchy is:

Figure
   ↓
Axes
   ↓
Artists
""")

# ==========================================================
# SAMPLE DATA
# ==========================================================

x = np.arange(1, 11)

y = np.array([3, 6, 4, 8, 7, 9, 11, 10, 12, 15])

# ==========================================================
# FIGURE & AXES
# ==========================================================

print("\nCreating Figure and Axes...")

fig, ax = plt.subplots(figsize=(8,5), dpi=120)

print("Figure Object :", fig)

print("Axes Object :", ax)

# ==========================================================
# BASIC LINE PLOT
# ==========================================================

ax.plot(
    x,
    y,
    marker="o",
    linewidth=2,
    label="Sales Trend"
)

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

ax.set_xlabel("Month")

ax.set_ylabel("Sales")

# ----------------------------------------------------------
# Title
# ----------------------------------------------------------

ax.set_title("Monthly Sales Trend")

# ----------------------------------------------------------
# Legend
# ----------------------------------------------------------

ax.legend()

# ----------------------------------------------------------
# Grid
# ----------------------------------------------------------

ax.grid(True)

# ----------------------------------------------------------
# Save Figure
# ----------------------------------------------------------

plt.savefig(
    "images/basic_line_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# BUSINESS INSIGHT
# ==========================================================

print("""

Business Insight
----------------

Sales generally increase over time.

There are small fluctuations,
but the overall trend is upward.

This indicates positive business growth.

""")

# ==========================================================
# MULTIPLE LINES
# ==========================================================

expenses = np.array([2,3,3,4,5,5,6,6,7,8])

plt.figure(figsize=(8,5))

plt.plot(
    x,
    y,
    marker="o",
    label="Sales"
)

plt.plot(
    x,
    expenses,
    marker="s",
    label="Expenses"
)

plt.title("Sales vs Expenses")

plt.xlabel("Month")

plt.ylabel("Amount")

plt.legend()

plt.grid(True)

plt.savefig(
    "images/multiple_lines.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# Business Insight
# ----------------------------------------------------------

print("""

Observation

Sales remain higher than expenses
throughout the year.

Profit appears positive every month.

""")

# ==========================================================
# FIGURE SIZE & DPI
# ==========================================================

print("\nFigure Size Used : (8,5)")

print("DPI Used : 300")

print("""

Higher DPI produces higher-quality images,
making them suitable for reports,
presentations, and publications.

""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "="*80)
print("MINI PRACTICE")
print("="*80)

print("""

Try the following:

1.

Create a line chart showing
student marks.

2.

Change the line color.

3.

Use different markers.

4.

Add a title.

5.

Add x and y labels.

6.

Save the figure.

7.

Increase the figure size.

8.

Change DPI to 600.

""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "="*80)
print("LESSON SUMMARY")
print("="*80)

print("""

Skills Learned

✔ Figure

✔ Axes

✔ Artist

✔ Line Plot

✔ Labels

✔ Title

✔ Legend

✔ Grid

✔ Figure Size

✔ DPI

✔ Saving Figures

""")

# ==========================================================
# LESSON COMPLETED
# ==========================================================

print("\n" + "="*80)
print("LESSON 01 COMPLETED SUCCESSFULLY")
print("="*80)
