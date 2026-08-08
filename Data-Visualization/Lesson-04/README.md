# Lesson 04 – Seaborn Statistical Visualization

## 📌 Objective

The objective of this lesson was to learn Seaborn, a high-level statistical visualization library built on top of Matplotlib.

Seaborn provides a simpler interface for creating informative and statistically meaningful visualizations, especially when working with Pandas DataFrames.

This lesson focuses on understanding different Seaborn plotting functions, selecting appropriate visualizations, customizing plots, and interpreting the resulting patterns.

---

## 📚 Topics Covered

### Introduction to Seaborn

- What is Seaborn?
- Seaborn vs Matplotlib
- Working with Pandas DataFrames
- Statistical visualization

### Relational Plots

- `relplot()`
- Scatter plots
- Line plots
- `hue`
- `style`
- `size`
- `col`
- `row`

### Categorical Plots

- `catplot()`
- Categorical scatter plots
- Box plots
- Violin plots
- Bar plots
- Count plots

### Distribution Plots

- `displot()`
- Histograms
- KDE plots

### Relationship Analysis

- `jointplot()`
- `pairplot()`

### Correlation Analysis

- `heatmap()`
- `clustermap()`

### Styling

- Seaborn themes
- Color palettes
- Figure customization
- Annotations
- Custom legends

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Create statistical visualizations using Seaborn.
- Choose appropriate plots for numerical and categorical variables.
- Analyze relationships between multiple features.
- Visualize distributions and correlations.
- Use `hue`, `style`, `size`, `row`, and `col`.
- Customize Seaborn visualizations.
- Add annotations and legends.
- Interpret statistical plots and derive insights.

---

## 📊 Dataset Used

Titanic Dataset

The Titanic dataset is useful for this lesson because it contains:

- Numerical variables
- Categorical variables
- Missing values
- Binary target variable
- Multiple relationships between features

---

## ⚡ Quick Revision

| Function | Purpose |
|----------|---------|
| `relplot()` | Relationship between variables |
| `catplot()` | Categorical data visualization |
| `displot()` | Distribution analysis |
| `jointplot()` | Relationship + distributions |
| `pairplot()` | Multiple pairwise relationships |
| `heatmap()` | Matrix/correlation visualization |
| `clustermap()` | Hierarchical clustering visualization |
| `set_theme()` | Set global Seaborn theme |
| `set_palette()` | Set color palette |

---

## 🔍 Choosing the Right Plot

### Relationship Between Numerical Variables

Use:

```python
sns.relplot()
sns.scatterplot()
sns.jointplot()
```

### Comparing Categories

Use:

```python
sns.catplot()
sns.boxplot()
sns.violinplot()
sns.barplot()
```

### Distribution

Use:

```python
sns.displot()
sns.histplot()
sns.kdeplot()
```

### Multiple Variables

Use:

```python
sns.pairplot()
```

### Correlation / Matrix

Use:

```python
sns.heatmap()
sns.clustermap()
```

---

## 📦 Libraries Used

- pandas
- seaborn
- matplotlib

Install Seaborn if required:

```bash
pip install seaborn
```

---

## 🚀 Key Takeaway

Seaborn makes statistical visualization easier by combining Pandas DataFrames with high-level plotting functions.

The important skill is not memorizing plotting functions, but understanding:

**Data Type → Question → Appropriate Visualization → Interpretation**

---

## Next Lesson

The next lesson will cover interactive visualization using Plotly, including interactive charts, Plotly Express, Plotly Graph Objects, 3D visualization, animation, and geographical visualization.
