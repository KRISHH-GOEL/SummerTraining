"""
Lesson 05 : Interactive Visualization with Plotly

Topics Covered
---------------
1. Plotly Introduction
2. Plotly Express
3. Interactive Line Chart
4. Interactive Bar Chart
5. Interactive Scatter Plot
6. Interactive Histogram
7. Interactive Box Plot
8. Interactive Pie Chart
9. Hover Information
10. Plotly Graph Objects
11. Multiple Traces
12. 3D Scatter Plot
13. Animated Visualization
14. Choropleth Map
15. Exporting Interactive HTML

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

import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# SETUP
# ==========================================================

os.makedirs("images", exist_ok=True)

print("=" * 80)
print("INTERACTIVE VISUALIZATION WITH PLOTLY")
print("=" * 80)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = sns.load_dataset("titanic")

print("\nDataset Loaded Successfully")

print("Shape :", df.shape)

print("\nFirst Five Rows")

print(df.head())

# ==========================================================
# 1. INTERACTIVE LINE CHART
# ==========================================================
# Line charts are useful for showing trends.
#
# Titanic does not contain a true time-series column,
# therefore we create a simple passenger index for
# demonstration purposes.

print("\n" + "=" * 80)
print("INTERACTIVE LINE CHART")
print("=" * 80)

line_data = (
    df.groupby("pclass", as_index=False)["fare"]
    .mean()
)

fig = px.line(
    line_data,
    x="pclass",
    y="fare",
    markers=True,
    title="Average Fare by Passenger Class",
    labels={
        "pclass": "Passenger Class",
        "fare": "Average Fare"
    }
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    "images/interactive_line_chart.html"
)

fig.show()

print("""
Insight
-------
Average fare increases substantially
as passenger class increases.
""")

# ==========================================================
# 2. INTERACTIVE BAR CHART
# ==========================================================

print("\n" + "=" * 80)
print("INTERACTIVE BAR CHART")
print("=" * 80)

bar_data = (
    df.groupby(
        ["class", "sex"],
        as_index=False
    )["survived"]
    .mean()
)

fig = px.bar(
    bar_data,
    x="class",
    y="survived",
    color="sex",
    barmode="group",
    title="Survival Rate by Class and Gender",
    labels={
        "class": "Passenger Class",
        "survived": "Survival Rate",
        "sex": "Gender"
    },
    hover_data=["survived"]
)

fig.update_yaxes(
    tickformat=".0%"
)

fig.write_html(
    "images/interactive_bar_chart.html"
)

fig.show()

print("""
Insight
-------
Survival rates vary substantially
across both passenger class and gender.
""")

# ==========================================================
# 3. INTERACTIVE SCATTER PLOT
# ==========================================================

print("\n" + "=" * 80)
print("INTERACTIVE SCATTER PLOT")
print("=" * 80)

scatter_data = df.dropna(
    subset=["age", "fare"]
)

fig = px.scatter(
    scatter_data,
    x="age",
    y="fare",
    color="survived",
    symbol="sex",
    size="pclass",
    hover_data=[
        "class",
        "embark_town"
    ],
    title="Age vs Fare",
    labels={
        "age": "Age",
        "fare": "Fare",
        "survived": "Survived"
    }
)

fig.write_html(
    "images/interactive_scatter.html"
)

fig.show()

print("""
Insight
-------
The interactive chart allows individual
passengers to be inspected using hover information.
""")

# ==========================================================
# 4. INTERACTIVE HISTOGRAM
# ==========================================================

print("\n" + "=" * 80)
print("INTERACTIVE HISTOGRAM")
print("=" * 80)

fig = px.histogram(
    df,
    x="age",
    color="sex",
    nbins=30,
    marginal="box",
    title="Age Distribution by Gender"
)

fig.write_html(
    "images/interactive_histogram.html"
)

fig.show()

print("""
Insight
-------
The histogram shows the distribution of passenger
ages while the marginal box plot helps identify
central tendency and potential outliers.
""")

# ==========================================================
# 5. INTERACTIVE BOX PLOT
# ==========================================================

print("\n" + "=" * 80)
print("INTERACTIVE BOX PLOT")
print("=" * 80)

fig = px.box(
    df,
    x="class",
    y="fare",
    color="sex",
    points="outliers",
    title="Fare Distribution by Class and Gender"
)

fig.write_html(
    "images/interactive_boxplot.html"
)

fig.show()

# ==========================================================
# 6. INTERACTIVE PIE CHART
# ==========================================================

print("\n" + "=" * 80)
print("INTERACTIVE PIE CHART")
print("=" * 80)

gender_counts = (
    df["sex"]
    .value_counts()
    .reset_index()
)

gender_counts.columns = [
    "sex",
    "count"
]

fig = px.pie(
    gender_counts,
    names="sex",
    values="count",
    title="Passenger Gender Distribution",
    hole=0.35
)

fig.write_html(
    "images/interactive_pie.html"
)

fig.show()

# ==========================================================
# 7. HOVER INFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("CUSTOM HOVER INFORMATION")
print("=" * 80)

hover_data = df.dropna(
    subset=["age", "fare"]
).head(200)

fig = px.scatter(
    hover_data,
    x="age",
    y="fare",
    color="class",
    hover_name="who",
    hover_data=[
        "sex",
        "survived",
        "embark_town"
    ],
    title="Passenger Details on Hover"
)

fig.write_html(
    "images/custom_hover.html"
)

fig.show()

print("""
Hover information allows users to inspect
individual observations without displaying
all information directly on the chart.
""")

# ==========================================================
# 8. PLOTLY GRAPH OBJECTS
# ==========================================================

print("\n" + "=" * 80)
print("PLOTLY GRAPH OBJECTS")
print("=" * 80)

class_data = (
    df.groupby("class")["fare"]
    .mean()
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=class_data.index.astype(str),
        y=class_data.values,
        name="Average Fare"
    )
)

fig.update_layout(
    title="Average Fare by Passenger Class",
    xaxis_title="Passenger Class",
    yaxis_title="Average Fare",
    template="plotly_white"
)

fig.write_html(
    "images/graph_objects_bar.html"
)

fig.show()

# ==========================================================
# 9. MULTIPLE TRACES
# ==========================================================

print("\n" + "=" * 80)
print("MULTIPLE TRACES")
print("=" * 80)

class_data = (
    df.groupby("class")
    .agg(
        Average_Fare=("fare", "mean"),
        Survival_Rate=("survived", "mean")
    )
    .reset_index()
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=class_data["class"],
        y=class_data["Average_Fare"],
        name="Average Fare"
    )
)

fig.add_trace(
    go.Scatter(
        x=class_data["class"],
        y=class_data["Survival_Rate"] * 100,
        mode="lines+markers",
        name="Survival Rate (%)"
    )
)

fig.update_layout(
    title="Fare and Survival Rate by Class",
    xaxis_title="Passenger Class",
    yaxis_title="Value"
)

fig.write_html(
    "images/multiple_traces.html"
)

fig.show()

# ==========================================================
# 10. 3D SCATTER PLOT
# ==========================================================

print("\n" + "=" * 80)
print("3D SCATTER PLOT")
print("=" * 80)

three_d_data = df.dropna(
    subset=[
        "age",
        "fare",
        "pclass"
    ]
)

fig = px.scatter_3d(
    three_d_data,
    x="age",
    y="fare",
    z="pclass",
    color="survived",
    symbol="sex",
    title="3D Passenger Visualization",
    labels={
        "age": "Age",
        "fare": "Fare",
        "pclass": "Passenger Class"
    }
)

fig.write_html(
    "images/3d_scatter.html"
)

fig.show()

# ==========================================================
# 11. ANIMATED CHART
# ==========================================================
# Animation requires a changing variable.
# We create age groups for demonstration.

print("\n" + "=" * 80)
print("ANIMATED VISUALIZATION")
print("=" * 80)

animation_data = df.dropna(
    subset=["age", "fare"]
).copy()

animation_data["Age_Group"] = (
    animation_data["age"] // 10 * 10
).astype(int)

animation_summary = (
    animation_data
    .groupby(
        ["Age_Group", "sex"],
        as_index=False
    )
    .agg(
        Average_Fare=("fare", "mean"),
        Passenger_Count=("fare", "count")
    )
)

fig = px.scatter(
    animation_summary,
    x="Average_Fare",
    y="Passenger_Count",
    color="sex",
    size="Passenger_Count",
    animation_frame="Age_Group",
    range_x=[
        0,
        animation_summary["Average_Fare"].max() * 1.2
    ],
    range_y=[
        0,
        animation_summary["Passenger_Count"].max() * 1.2
    ],
    title="Passenger Distribution Across Age Groups"
)

fig.write_html(
    "images/animated_chart.html"
)

fig.show()

# ==========================================================
# 12. CHOROPLETH MAP
# ==========================================================
# Titanic does not contain country-level data,
# so a small demonstration dataset is created.

print("\n" + "=" * 80)
print("CHOROPLETH MAP")
print("=" * 80)

countries = pd.DataFrame({

    "Country": [
        "India",
        "United States",
        "United Kingdom",
        "Canada",
        "Australia",
        "Germany",
        "France"
    ],

    "Value": [
        85,
        75,
        65,
        55,
        70,
        60,
        68
    ]

})

fig = px.choropleth(
    countries,
    locations="Country",
    locationmode="country names",
    color="Value",
    title="Example Global Data Distribution",
    color_continuous_scale="Viridis"
)

fig.write_html(
    "images/choropleth_map.html"
)

fig.show()

print("""
Choropleth maps are useful when the geographical
location of an observation is important.

Examples:
- Population
- Sales
- GDP
- Disease Rates
- Customer Distribution
""")

# ==========================================================
# 13. EXPORTING INTERACTIVE VISUALIZATIONS
# ==========================================================

print("\n" + "=" * 80)
print("EXPORTING INTERACTIVE VISUALIZATIONS")
print("=" * 80)

export_file = "images/final_interactive_chart.html"

fig = px.scatter(
    df.dropna(subset=["age", "fare"]),
    x="age",
    y="fare",
    color="class",
    title="Final Interactive Titanic Visualization"
)

fig.write_html(export_file)

print(
    f"Interactive visualization saved to: {export_file}"
)

# ==========================================================
# PLOTLY EXPRESS VS GRAPH OBJECTS
# ==========================================================

print("\n" + "=" * 80)
print("PLOTLY EXPRESS VS GRAPH OBJECTS")
print("=" * 80)

comparison = pd.DataFrame({

    "Plotly Express": [
        "Quick",
        "Concise",
        "Easy to learn",
        "Ideal for standard charts",
        "Excellent for EDA"
    ],

    "Graph Objects": [
        "More control",
        "More code",
        "More complex",
        "Ideal for custom figures",
        "Excellent for advanced dashboards"
    ]

})

print(comparison.to_string(index=False))

# ==========================================================
# VISUALIZATION SELECTION GUIDE
# ==========================================================

print("\n" + "=" * 80)
print("VISUALIZATION SELECTION GUIDE")
print("=" * 80)

print("""
Interactive Visualization
--------------------------

Trend
-----
px.line()

Category Comparison
-------------------
px.bar()

Relationship
------------
px.scatter()

Distribution
------------
px.histogram()
px.box()

Composition
-----------
px.pie()

Geographical Data
-----------------
px.choropleth()

3D Relationships
----------------
px.scatter_3d()

Changing Data Over Time
-----------------------
animation_frame

Advanced Customization
----------------------
plotly.graph_objects
""")

# ==========================================================
# FINAL BUSINESS SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("""
Key Findings
------------

1. Plotly provides interactive alternatives
   to traditional static visualizations.

2. Hover information allows detailed
   exploration of individual observations.

3. Plotly Express is useful for rapid
   exploratory visualization.

4. Graph Objects provides greater control
   over complex visualizations.

5. 3D charts can represent relationships
   involving three numerical dimensions.

6. Animated charts can represent changes
   across a fourth dimension.

7. Choropleth maps are useful for
   geographically distributed data.

8. Interactive HTML files can be embedded
   into web applications and reports.
""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Create an interactive line chart.

2. Create an interactive bar chart.

3. Add custom hover information.

4. Create a Plotly Express scatter plot.

5. Recreate the scatter plot using
   Graph Objects.

6. Create a 3D visualization.

7. Create an animated visualization.

8. Create a choropleth map using
   a real geographical dataset.

9. Export all charts as HTML.

10. Compare Plotly with Matplotlib
    and Seaborn.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Plotly Express

✔ Plotly Graph Objects

✔ Interactive Charts

✔ Hover Information

✔ Multiple Traces

✔ 3D Scatter Plots

✔ Animated Charts

✔ Choropleth Maps

✔ HTML Export

✔ Interactive Data Exploration
""")

print("\nLesson 05 Completed Successfully!")
