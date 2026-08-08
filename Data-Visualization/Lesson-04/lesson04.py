"""
Lesson 04 : Seaborn Statistical Visualization

Topics Covered
---------------
1. Seaborn Introduction
2. relplot()
3. catplot()
4. displot()
5. jointplot()
6. pairplot()
7. heatmap()
8. clustermap()
9. Themes
10. Color Palettes
11. Annotations
12. Custom Legends

Dataset
-------
Titanic Dataset

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# SETUP
# ==========================================================

os.makedirs("images", exist_ok=True)

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

print("=" * 80)
print("SEABORN STATISTICAL VISUALIZATION")
print("=" * 80)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = sns.load_dataset("titanic")

print("\nDataset Loaded Successfully")

print("\nShape :", df.shape)

print("\nColumns")

print(df.columns.tolist())

# ==========================================================
# BASIC INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("BASIC DATA INSPECTION")
print("=" * 80)

print(df.head())

print("\nData Types")

print(df.dtypes)

print("\nMissing Values")

print(df.isnull().sum())

# ==========================================================
# 1. RELPLOT
# ==========================================================
# relplot() is used to visualize relationships between
# numerical variables.
#
# hue allows us to represent another variable using
# different groups.

print("\n" + "=" * 80)
print("RELPLOT")
print("=" * 80)

rel_plot = sns.relplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    height=6,
    aspect=1.3
)

rel_plot.set(
    title="Age vs Fare by Survival and Gender",
    xlabel="Age",
    ylabel="Fare"
)

rel_plot.savefig(
    "images/relplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
Fare varies considerably among passengers.

The plot also shows differences in fare and age
between survivors and non-survivors.
""")

# ==========================================================
# 2. CATPLOT
# ==========================================================
# catplot() provides a unified interface for
# categorical visualizations.

print("\n" + "=" * 80)
print("CATPLOT")
print("=" * 80)

cat_plot = sns.catplot(
    data=df,
    x="class",
    y="fare",
    hue="sex",
    kind="box",
    height=6,
    aspect=1.3
)

cat_plot.set(
    title="Fare Distribution by Passenger Class and Gender",
    xlabel="Passenger Class",
    ylabel="Fare"
)

cat_plot.savefig(
    "images/catplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
Fare distributions differ substantially across
passenger classes.

First-class passengers show much higher fares.
""")

# ==========================================================
# 3. DISPlot
# ==========================================================
# displot() is designed for distribution analysis.

print("\n" + "=" * 80)
print("DISPLOT")
print("=" * 80)

dis_plot = sns.displot(
    data=df,
    x="age",
    hue="sex",
    kde=True,
    height=6,
    aspect=1.3
)

dis_plot.set(
    title="Age Distribution by Gender"
)

dis_plot.savefig(
    "images/displot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
Most passengers fall within the young-adult
to middle-aged range.

The age distribution differs somewhat between
male and female passengers.
""")

# ==========================================================
# 4. JOINTPLOT
# ==========================================================
# jointplot() displays the relationship between
# two variables together with their individual
# distributions.

print("\n" + "=" * 80)
print("JOINTPLOT")
print("=" * 80)

joint = sns.jointplot(
    data=df,
    x="age",
    y="fare",
    kind="scatter",
    hue="survived",
    height=7
)

joint.fig.suptitle(
    "Age vs Fare with Survival",
    y=1.02
)

joint.savefig(
    "images/jointplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
The joint distribution shows that fare is
concentrated at lower values, with a small
number of high-fare observations.
""")

# ==========================================================
# 5. PAIRPLOT
# ==========================================================
# pairplot() compares several numerical variables
# against each other.

print("\n" + "=" * 80)
print("PAIRPLOT")
print("=" * 80)

pair_data = df[
    [
        "age",
        "fare",
        "pclass",
        "sibsp",
        "parch",
        "survived"
    ]
].dropna()

pair = sns.pairplot(
    pair_data,
    hue="survived",
    diag_kind="hist"
)

pair.fig.suptitle(
    "Pairwise Relationships in Titanic Dataset",
    y=1.02
)

pair.savefig(
    "images/pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
Pairplot provides a broad view of relationships
between multiple numerical variables.

It can help identify:
- Correlations
- Clusters
- Outliers
- Potentially useful features
""")

# ==========================================================
# 6. HEATMAP
# ==========================================================
# Heatmaps are useful for visualizing matrices,
# particularly correlation matrices.

print("\n" + "=" * 80)
print("HEATMAP")
print("=" * 80)

numeric_df = df.select_dtypes(
    include="number"
)

correlation = numeric_df.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar=True
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "images/heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
The correlation matrix helps identify
linear relationships between numerical features.

Correlation should be interpreted carefully
and does not imply causation.
""")

# ==========================================================
# 7. CLUSTERMAP
# ==========================================================
# clustermap() combines a heatmap with hierarchical
# clustering.

print("\n" + "=" * 80)
print("CLUSTERMAP")
print("=" * 80)

cluster_data = numeric_df.dropna()

cluster = sns.clustermap(
    cluster_data.corr(),
    annot=True,
    fmt=".2f",
    figsize=(9, 8)
)

cluster.fig.suptitle(
    "Hierarchical Clustering of Correlations",
    y=1.02
)

cluster.savefig(
    "images/clustermap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Insight
-------
Clustermap groups variables with similar
correlation patterns.

This can help identify related features
during exploratory analysis.
""")

# ==========================================================
# 8. THEME CUSTOMIZATION
# ==========================================================

print("\n" + "=" * 80)
print("SEABORN THEMES")
print("=" * 80)

sns.set_theme(
    style="darkgrid",
    context="notebook"
)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="class"
)

plt.title("Passenger Count by Class")

plt.tight_layout()

plt.savefig(
    "images/theme_example.png",
    dpi=300
)

plt.show()

# Reset theme for subsequent plots.

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

# ==========================================================
# 9. COLOR PALETTE
# ==========================================================

print("\n" + "=" * 80)
print("COLOR PALETTES")
print("=" * 80)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="class",
    hue="sex",
    palette="deep"
)

plt.title("Passenger Class by Gender")

plt.tight_layout()

plt.savefig(
    "images/color_palette.png",
    dpi=300
)

plt.show()

print("""
Color palettes should be chosen based on
the purpose of the visualization.

Avoid using excessive colors when they
do not communicate additional information.
""")

# ==========================================================
# 10. ANNOTATIONS
# ==========================================================

print("\n" + "=" * 80)
print("ANNOTATIONS")
print("=" * 80)

survival_rate = (
    df.groupby("class")["survived"]
    .mean()
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=survival_rate.index,
    y=survival_rate.values
)

plt.title("Survival Rate by Passenger Class")

plt.xlabel("Passenger Class")

plt.ylabel("Survival Rate")

# Add values above each bar.

for index, value in enumerate(survival_rate.values):

    ax.text(
        index,
        value + 0.02,
        f"{value:.2%}",
        ha="center",
        fontsize=10
    )

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "images/annotated_barplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("""
Annotation Insight
------------------
Annotations make important numerical values
immediately visible to the reader.
""")

# ==========================================================
# 11. CUSTOM LEGEND
# ==========================================================

print("\n" + "=" * 80)
print("CUSTOM LEGEND")
print("=" * 80)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    size="pclass"
)

plt.title(
    "Age vs Fare by Survival, Gender and Class"
)

plt.xlabel("Age")

plt.ylabel("Fare")

plt.legend(
    title="Passenger Information",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "images/custom_legend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# FINAL COMPARISON
# ==========================================================

print("\n" + "=" * 80)
print("SEABORN PLOT SELECTION GUIDE")
print("=" * 80)

plot_guide = pd.DataFrame({

    "Question": [
        "How are two numerical variables related?",
        "How do categories compare?",
        "What does a numerical distribution look like?",
        "How do multiple numerical variables relate?",
        "How strongly are variables correlated?",
        "Which variables have similar correlation patterns?"
    ],

    "Recommended Plot": [
        "relplot / scatterplot",
        "catplot",
        "displot",
        "pairplot",
        "heatmap",
        "clustermap"
    ]
})

print(plot_guide.to_string(index=False))

# ==========================================================
# FINAL BUSINESS SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("FINAL EDA SUMMARY")
print("=" * 80)

print("""
Key Findings
------------

1. Passenger class is strongly associated with fare.

2. Survival rates differ significantly across
   passenger classes.

3. Fare has a skewed distribution with
   high-value outliers.

4. Age and fare show substantial variation
   across passengers.

5. Gender provides useful information when
   analyzing survival.

6. Correlation analysis can help identify
   relationships between numerical features.

7. Seaborn makes it easier to combine
   statistical information with visualizations.

""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Use relplot() to analyze:
   age vs fare by passenger class.

2. Use catplot() to compare survival
   across gender and class.

3. Use displot() to analyze fare distribution.

4. Use jointplot() to investigate
   age vs fare.

5. Create a pairplot using:
   age, fare, pclass and survived.

6. Create a heatmap for numerical features.

7. Create a clustermap using the
   correlation matrix.

8. Change the Seaborn theme.

9. Experiment with different palettes.

10. Add annotations to a bar chart.

11. Create a custom legend.

12. Write five business insights
    from your visualizations.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ relplot()

✔ catplot()

✔ displot()

✔ jointplot()

✔ pairplot()

✔ heatmap()

✔ clustermap()

✔ Themes

✔ Color Palettes

✔ Annotations

✔ Custom Legends

✔ Statistical Visualization

✔ Business Interpretation
""")

print("\nLesson 04 Completed Successfully!")
