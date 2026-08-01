# Lesson 04 – Outlier Detection & Data Transformation

## 📌 Objective

The objective of this lesson was to understand how to identify, analyze, and treat outliers in a dataset. Outliers are observations that significantly differ from the rest of the data and can negatively impact statistical analysis and machine learning models.

This lesson also introduces numerical data transformation techniques used to reduce skewness and improve model performance.

---

## 📚 Topics Covered

### What are Outliers?

- Definition of Outliers
- Causes of Outliers
- Effect of Outliers on Machine Learning

### Outlier Detection Techniques

- Z-Score Method
- Interquartile Range (IQR) Method
- Isolation Forest

### Outlier Treatment

- Removing Outliers
- Winsorization (Capping Extreme Values)

### Data Transformation

- Log Transformation
- Box-Cox Transformation
- Yeo-Johnson Transformation

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Detect outliers using statistical methods.
- Visualize outliers using boxplots.
- Remove or cap extreme values.
- Apply Winsorization.
- Detect anomalies using Isolation Forest.
- Reduce skewness using transformation techniques.

---

## 📝 Dataset Used

Titanic Dataset (Seaborn)

---

## ⚡ Quick Revision

| Method | Purpose |
|---------|---------|
| Z-Score | Detect values beyond a threshold (typically ±3) |
| IQR | Detect outliers using quartiles |
| Winsorization | Cap extreme values |
| Isolation Forest | Detect anomalies using ML |
| np.log1p() | Log Transformation |
| PowerTransformer(Box-Cox) | Normalize positive data |
| PowerTransformer(Yeo-Johnson) | Normalize positive & negative data |

---

## 📦 Libraries Used

- pandas
- numpy
- seaborn
- matplotlib
- scipy
- scikit-learn

---

## 🚀 Next Lesson

In the next lesson, I will learn Feature Engineering techniques including encoding, scaling, polynomial features, transformations, and handling imbalanced datasets.
