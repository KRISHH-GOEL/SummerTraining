# Lesson 06 – Dash Analytics Dashboard

## 📌 Objective

The objective of this lesson was to learn the fundamentals of Dash and understand how interactive data visualizations can be combined into a web-based analytics dashboard using Python.

Dash is a Python framework built around Plotly that allows data scientists to create interactive analytical applications without requiring a separate JavaScript frontend.

This lesson combines:

- Pandas for data manipulation
- Plotly for visualization
- Dash for application development

---

## 📚 Topics Covered

### Introduction to Dash

- What is Dash?
- Dash application architecture
- Dash vs Plotly
- Dash vs traditional web development

### Dash Application Structure

- `Dash()`
- `app.layout`
- Components
- Callbacks

### Dash HTML Components

- `html.Div`
- `html.H1`
- `html.H2`
- `html.P`

### Dash Core Components

- `dcc.Graph`
- `dcc.Dropdown`
- `dcc.Slider`

### Interactive Callbacks

- `@app.callback`
- Inputs
- Outputs
- Updating graphs dynamically

### Dashboard Development

- Creating KPI cards
- Creating interactive charts
- Adding filters
- Connecting multiple charts
- Dashboard layout

### Running the Application

```bash
python lesson06_dash_dashboard.py
```

Then open:

```text
http://127.0.0.1:8050/
```

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Create a basic Dash application.
- Build layouts using Dash components.
- Embed Plotly visualizations into Dash.
- Create dropdown filters.
- Create interactive callbacks.
- Dynamically update charts.
- Build a basic analytics dashboard.

---

## 🧱 Dash Architecture

A basic Dash application follows this structure:

```text
Python Data
     ↓
Pandas
     ↓
Plotly Figure
     ↓
Dash Layout
     ↓
Dash Components
     ↓
Callback
     ↓
Interactive Dashboard
```

---

## 🔄 Callback Concept

Dash callbacks connect user interactions with application outputs.

Example:

```python
@app.callback(
    Output("graph", "figure"),
    Input("dropdown", "value")
)
def update_graph(selected_value):

    filtered_data = df[
        df["class"] == selected_value
    ]

    return px.histogram(filtered_data)
```

The workflow is:

```text
User changes Dropdown
        ↓
Input detected
        ↓
Callback executes
        ↓
Data is filtered
        ↓
New Plotly Figure created
        ↓
Graph updates
```

---

## 📊 Dashboard Features

The dashboard developed in this lesson contains:

- Total Passengers
- Survival Rate
- Average Fare
- Passenger Class Filter
- Gender Filter
- Survival Distribution
- Fare Distribution
- Age Distribution
- Class-wise Survival Analysis

---

## 📦 Libraries Used

- pandas
- seaborn
- plotly
- dash

Install them using:

```bash
pip install pandas seaborn plotly dash
```

---

## 🚀 Running the Application

Navigate to the lesson directory and execute:

```bash
python lesson06_dash_dashboard.py
```

The terminal will provide a local URL.

Open the URL in a browser:

```text
http://127.0.0.1:8050/
```

Use the dropdown filters to interact with the dashboard.

---

## 🧠 Key Takeaway

Dash allows Python-based data analysis and interactive visualization to be transformed into an actual web application.

The basic workflow is:

```text
Data
 ↓
Analysis
 ↓
Plotly Visualization
 ↓
Dash Layout
 ↓
Callbacks
 ↓
Interactive Dashboard
```

---

## Next Lesson

The next lesson will introduce **Power BI for Data Science**, including data connections, Power Query, DAX, dashboard visualizations, KPIs, slicers, and publishing dashboards.
