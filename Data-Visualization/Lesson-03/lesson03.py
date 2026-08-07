"""
Lesson 03 : Advanced Matplotlib

Topics Covered
---------------
1. Figure Customization
2. Subplots
3. Shared Axes
4. Twin Axes
5. Annotation
6. Text
7. Axis Limits
8. Figure Export

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
print("ADVANCED MATPLOTLIB")
print("=" * 80)

df = sns.load_dataset("titanic")

# ==========================================================
# FIGURE CUSTOMIZATION
# ==========================================================

plt.figure(figsize=(10,6), dpi=150)

fare = (
    df.groupby("class")["fare"]
    .mean()
)

plt.bar(
    fare.index.astype(str),
    fare.values
)

plt.title("Average Fare by Passenger Class",
          fontsize=16)

plt.xlabel("Passenger Class")

plt.ylabel("Average Fare")

plt.grid(axis="y", alpha=0.3)

plt.savefig(
    "images/customized_bar_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Business Insight
----------------
Average fare increases significantly
from Third Class to First Class.
""")

# ==========================================================
# SUBPLOTS
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12,5)
)

# Histogram

axes[0].hist(
    df["age"].dropna(),
    bins=20
)

axes[0].set_title("Age Distribution")

# Boxplot

axes[1].boxplot(
    df["fare"].dropna()
)

axes[1].set_title("Fare Distribution")

plt.tight_layout()

plt.savefig(
    "images/subplots.png",
    dpi=300
)

plt.show()

# ==========================================================
# SHARED AXES
# ==========================================================

fig, ax = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(8,6)
)

ax[0].hist(df["age"].dropna())

ax[0].set_title("Age")

ax[1].hist(df["fare"].dropna())

ax[1].set_title("Fare")

plt.tight_layout()

plt.savefig(
    "images/shared_axes.png",
    dpi=300
)

plt.show()

# ==========================================================
# TWIN AXES
# ==========================================================

monthly = pd.DataFrame({

    "Month":[1,2,3,4,5,6],

    "Sales":[25,30,35,40,42,50],

    "Profit":[5,6,8,10,11,13]

})

fig, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(
    monthly["Month"],
    monthly["Sales"],
    color="blue",
    marker="o"
)

ax1.set_ylabel("Sales")

ax2 = ax1.twinx()

ax2.plot(
    monthly["Month"],
    monthly["Profit"],
    color="red",
    marker="s"
)

ax2.set_ylabel("Profit")

plt.title("Sales vs Profit")

plt.savefig(
    "images/twin_axes.png",
    dpi=300
)

plt.show()

print("""
Business Insight
----------------
Sales and profit increase together,
indicating healthy business growth.
""")

# ==========================================================
# ANNOTATIONS
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    monthly["Month"],
    monthly["Sales"],
    marker="o"
)

highest = monthly["Sales"].max()

highest_month = monthly.loc[
    monthly["Sales"].idxmax(),
    "Month"
]

plt.annotate(

    "Highest Sales",

    xy=(highest_month, highest),

    xytext=(4,55),

    arrowprops=dict(
        arrowstyle="->"
    )

)

plt.title("Sales Trend")

plt.grid(True)

plt.savefig(
    "images/annotation.png",
    dpi=300
)

plt.show()

# ==========================================================
# TEXT
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    fare.index.astype(str),
    fare.values
)

for i, value in enumerate(fare.values):

    plt.text(
        i,
        value+2,
        f"{value:.1f}",
        ha="center"
    )

plt.title("Average Fare")

plt.savefig(
    "images/text_labels.png",
    dpi=300
)

plt.show()

# ==========================================================
# AXIS LIMITS
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["age"].dropna(),
    bins=20
)

plt.xlim(0,80)

plt.ylim(0,200)

plt.title("Age Distribution with Axis Limits")

plt.savefig(
    "images/axis_limits.png",
    dpi=300
)

plt.show()

# ==========================================================
# PROFESSIONAL EXPORT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    monthly["Month"],
    monthly["Profit"],
    marker="o"
)

plt.title("Monthly Profit")

plt.tight_layout()

plt.savefig(

    "images/high_quality_plot.png",

    dpi=600,

    bbox_inches="tight"

)

plt.show()

print("""
Figure exported successfully
at publication quality (600 DPI).
""")

# ==========================================================
# BEST PRACTICES
# ==========================================================

print("\n" + "="*80)
print("BEST PRACTICES")
print("="*80)

print("""

✓ Always use titles.

✓ Label both axes.

✓ Use legends when plotting
multiple series.

✓ Avoid excessive colors.

✓ Keep charts uncluttered.

✓ Export at 300–600 DPI
for reports.

✓ Use tight_layout().

""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "="*80)
print("MINI PRACTICE")
print("="*80)

print("""

1.
Create two subplots.

2.
Annotate the highest value.

3.
Use twin axes.

4.
Export at 600 DPI.

5.
Customize axis limits.

6.
Add value labels on bars.

""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "="*80)
print("LESSON SUMMARY")
print("="*80)

print("""

Skills Learned

✔ Subplots

✔ Shared Axes

✔ Twin Axes

✔ Figure Customization

✔ Annotation

✔ Text Labels

✔ Axis Limits

✔ Professional Figure Export

""")

print("\nLesson 03 Completed Successfully.")
