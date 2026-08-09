# Lesson 05 – Interactive Visualization with Plotly

## 📌 Objective

The objective of this lesson was to learn interactive data visualization using Plotly.

Unlike Matplotlib and Seaborn, which primarily produce static visualizations, Plotly allows users to interact directly with charts through features such as hovering, zooming, panning, filtering, and selecting data points.

This lesson introduces Plotly Express for quick visualization and Plotly Graph Objects for more detailed customization.

---

## 📚 Topics Covered

### Introduction to Plotly

- What is Plotly?
- Static vs Interactive Visualization
- Plotly Express
- Plotly Graph Objects

### Plotly Express

- Interactive Line Charts
- Interactive Bar Charts
- Interactive Scatter Plots
- Histograms
- Box Plots
- Pie Charts

### Interactivity

- Hover information
- Zoom
- Pan
- Legend interaction
- Filtering categories
- Custom hover data

### Plotly Graph Objects

- Figure
- Traces
- Layout
- Customization
- Adding multiple traces

### Advanced Visualizations

- 3D Scatter Plot
- Animated Charts
- Choropleth Maps

### Exporting Visualizations

- HTML files
- Interactive reports

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Create interactive visualizations using Plotly.
- Use Plotly Express for rapid visualization.
- Customize charts using Graph Objects.
- Add multiple traces to a figure.
- Create interactive 3D visualizations.
- Create animated charts.
- Create geographical visualizations.
- Export interactive charts as HTML files.

---

## 📊 Dataset Used

Titanic Dataset

Additional datasets are created where a specific Plotly visualization requires geographic, temporal, or 3D data.

---

## ⚡ Quick Revision

| Function / Concept | Purpose |
|---|---|
| `px.line()` | Interactive line chart |
| `px.bar()` | Interactive bar chart |
| `px.scatter()` | Interactive scatter plot |
| `px.histogram()` | Interactive histogram |
| `px.box()` | Interactive box plot |
| `px.pie()` | Interactive pie chart |
| `px.scatter_3d()` | 3D scatter visualization |
| `px.choropleth()` | Geographic visualization |
| `px.scatter(..., animation_frame=...)` | Animated visualization |
| `go.Figure()` | Create custom figure |
| `go.Scatter()` | Create scatter trace |
| `go.Bar()` | Create bar trace |
| `fig.update_layout()` | Customize figure |
| `fig.write_html()` | Save interactive chart |

---

## 🆚 Plotly Express vs Graph Objects

### Plotly Express

Best for:

- Quick visualizations
- Exploratory Data Analysis
- Standard charts
- Less code

### Graph Objects

Best for:

- Detailed customization
- Multiple traces
- Complex figures
- Advanced dashboards

---

## 📦 Libraries Used

- pandas
- plotly
- seaborn

Install Plotly:

```bash
pip install plotly
```

---

## 🚀 Key Takeaway

Plotly is particularly useful when users need to interact with visualizations rather than simply view static charts.

A typical workflow is:

```text
Load Data
    ↓
Prepare Data
    ↓
Choose Visualization
    ↓
Create Interactive Plot
    ↓
Customize
    ↓
Add Hover Information
    ↓
Export / Embed
```

---

## Next Lesson

The next lesson will introduce **Dash**, Plotly's Python framework for building interactive analytical dashboards.
