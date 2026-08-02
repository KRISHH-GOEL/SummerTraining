# Lesson 05 – Feature Engineering

## 📌 Objective

The objective of this lesson was to understand how to transform raw data into meaningful features that improve the performance of machine learning models.

Feature Engineering is one of the most important stages of the Machine Learning pipeline. Well-engineered features often have a greater impact on model performance than the choice of algorithm.

---

## 📚 Topics Covered

### Categorical Encoding

- Label Encoding
- One-Hot Encoding
- Target Encoding (Concept)
- Frequency Encoding

### Feature Scaling

- Min-Max Scaling
- Standardization (Z-Score Scaling)
- Robust Scaling

### Feature Transformation

- Equal Width Binning
- Equal Frequency Binning
- Polynomial Features
- Interaction Features
- Log Transformation

### Domain Feature Engineering

- Creating New Features
- Business-Oriented Features

### Handling Imbalanced Data

- Random Under Sampling
- SMOTE
- ADASYN

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Convert categorical variables into numerical representations.
- Scale numerical features using different scaling techniques.
- Create new informative features.
- Generate polynomial and interaction features.
- Apply binning techniques.
- Handle imbalanced datasets using sampling methods.

---

## 📝 Dataset Used

Titanic Dataset (Seaborn)

---

## ⚡ Quick Revision

| Technique | Purpose |
|-----------|---------|
| LabelEncoder | Encode categories as integers |
| OneHotEncoder / get_dummies | Create binary columns |
| Frequency Encoding | Replace categories with frequency |
| MinMaxScaler | Scale to 0–1 |
| StandardScaler | Mean = 0, Std = 1 |
| RobustScaler | Resistant to outliers |
| pd.cut() | Equal-width bins |
| pd.qcut() | Equal-frequency bins |
| PolynomialFeatures | Create polynomial terms |
| SMOTE | Oversample minority class |
| ADASYN | Adaptive synthetic oversampling |

---

## 📦 Libraries Used

- pandas
- numpy
- seaborn
- scikit-learn
- imbalanced-learn

Install additional package:

```bash
pip install imbalanced-learn
```

---

## 🚀 Next Lesson

In the next lesson, I will perform Exploratory Data Analysis (EDA) using statistical summaries and visualizations to understand patterns, relationships, and trends in the data.
