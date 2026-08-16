# Lesson 03 – Feature Engineering for Machine Learning

## 📌 Objective

The objective of this lesson was to understand how raw features can be transformed, created, selected, and represented in a way that helps Machine Learning models learn useful patterns.

Feature engineering is one of the most important parts of a Machine Learning workflow.

A good model trained on meaningful features can often outperform a more complex model trained on poorly represented data.

---

## 📚 Topics Covered

### Feature Engineering Fundamentals

- What is a feature?
- What is feature engineering?
- Feature creation
- Feature transformation
- Domain-specific features
- Feature representation

### Categorical Feature Encoding

- One-Hot Encoding
- Label Encoding
- Ordinal Encoding
- Frequency Encoding
- Target Encoding
- Choosing an appropriate encoding strategy

### Numerical Transformations

- Log transformation
- Power transformations
- Box-Cox transformation
- Yeo-Johnson transformation

### Binning / Discretization

- Continuous to categorical conversion
- Equal-width binning
- Equal-frequency binning
- `pd.cut()`
- `pd.qcut()`

### Polynomial Features

- Polynomial features
- Interaction terms
- `PolynomialFeatures`
- Feature expansion

### Feature Selection

- Why feature selection is important
- Correlation-based selection
- `SelectKBest`
- Mutual information
- Recursive Feature Elimination
- RFE concept

### Imbalanced Classification

- Class imbalance
- Majority class
- Minority class
- Random oversampling
- Random undersampling
- SMOTE
- ADASYN

### Important Practical Concepts

- Feature leakage
- Target leakage
- Feature selection before/after splitting
- Feature engineering inside pipelines

---

# 🧠 What Is Feature Engineering?

Feature engineering is the process of creating or transforming input variables so that they provide useful information to a Machine Learning model.

Example:

Raw features:

```text
Date
TotalCharges
Tenure
```

Possible engineered features:

```text
Year
Month
AverageMonthlySpend
TenureGroup
```

The model can then work with information that is more directly related to the underlying problem.

---

# 🔤 Encoding Categorical Features

Machine Learning algorithms generally require numerical representations.

### One-Hot Encoding

Example:

```text
City

Delhi
Mumbai
Chennai
```

becomes:

```text
City_Delhi
City_Mumbai
City_Chennai
```

Best for nominal categories.

---

### Label Encoding

Example:

```text
Low    → 0
Medium → 1
High   → 2
```

Appropriate when the categories have meaningful order, or commonly for encoding a classification target.

---

### Ordinal Encoding

Used when categories have an explicit order.

Example:

```text
Low < Medium < High
```

---

### Frequency Encoding

Each category is replaced by its frequency.

Example:

```text
Delhi   → 0.50
Mumbai  → 0.30
Chennai → 0.20
```

Useful when a categorical variable has many unique categories.

---

### Target Encoding

Each category is represented using a statistic of the target.

For example, in a binary classification problem:

```text
Delhi   → 0.72
Mumbai  → 0.55
Chennai → 0.31
```

These values represent the target rate for each category.

Target encoding must be performed carefully because it can cause target leakage.

For practical projects, use cross-validated target encoding or a leakage-safe implementation.

---

# 📊 Binning

Binning converts a continuous numerical feature into discrete groups.

Example:

```text
Age
 ↓
0–18       → Child
19–35      → Young Adult
36–60      → Adult
61+        → Senior
```

### Equal-Width Binning

Each interval has approximately the same numerical width.

```python
pd.cut()
```

### Equal-Frequency Binning

Each bin contains approximately the same number of observations.

```python
pd.qcut()
```

---

# 🔢 Polynomial Features

Polynomial features create additional features from existing numerical variables.

Suppose:

```text
X1
X2
```

Polynomial expansion can create:

```text
X1
X2
X1²
X2²
X1 × X2
```

This allows models such as Linear Regression to capture nonlinear relationships.

---

# 🔄 Log Transformation

A log transformation can reduce strong right skew.

Example:

```python
np.log1p(x)
```

`log1p(x)` is useful when zero values may be present because it computes:

```text
log(1 + x)
```

---

# 📐 Power Transformations

Power transformations can make numerical features more suitable for models that benefit from approximately symmetric distributions.

### Box-Cox

Requires strictly positive values.

### Yeo-Johnson

Can handle zero and negative values.

Scikit-learn provides:

```python
PowerTransformer()
```

---

# 🎯 Feature Selection

Feature selection means choosing the most useful features while removing unnecessary or redundant ones.

Benefits:

- Reduced dimensionality
- Faster training
- Lower complexity
- Reduced noise
- Potentially better generalization
- Improved interpretability

Methods covered:

```text
Correlation
     ↓
SelectKBest
     ↓
Mutual Information
     ↓
RFE
```

---

# ⚖️ Imbalanced Classification

A dataset is imbalanced when classes are represented very unevenly.

Example:

```text
Class 0 → 950 observations
Class 1 → 50 observations
```

A model that always predicts Class 0 would achieve:

```text
95% Accuracy
```

but would completely fail to identify the minority class.

Therefore, accuracy alone can be misleading.

---

# 🔄 SMOTE

SMOTE stands for:

**Synthetic Minority Over-sampling Technique**

Instead of simply duplicating minority observations, SMOTE generates synthetic minority samples based on neighboring observations.

General idea:

```text
Minority Samples
      ↓
Find Neighbors
      ↓
Generate Synthetic Samples
      ↓
More Balanced Dataset
```

---

# 🔄 ADASYN

ADASYN stands for:

**Adaptive Synthetic Sampling**

It generates more synthetic samples in regions where minority observations are harder to learn.

Conceptually:

```text
Easy Minority Samples
       ↓
Fewer Synthetic Samples

Hard Minority Samples
       ↓
More Synthetic Samples
```

---

# ⚠️ Important: Avoid Data Leakage

Resampling must **not** be performed before the train-test split.

Incorrect:

```text
Entire Dataset
      ↓
SMOTE
      ↓
Train-Test Split
```

Correct:

```text
Entire Dataset
      ↓
Train-Test Split
      ↓
SMOTE on Training Data Only
      ↓
Train Model
      ↓
Evaluate on Original Test Data
```

For robust workflows, use an `imblearn.pipeline.Pipeline` so resampling occurs correctly during training and cross-validation.

---

# 🧩 Feature Engineering Pipeline

```text
Raw Data
   ↓
Train-Test Split
   ↓
Feature Creation
   ↓
Feature Transformation
   ↓
Encoding
   ↓
Scaling
   ↓
Feature Selection
   ↓
Class Balancing
   ↓
Model Training
```

The exact order depends on the problem and the transformation being used.

---

# 📦 Libraries Used

- pandas
- NumPy
- Scikit-learn
- imbalanced-learn

Install:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---

# 🧪 Dataset Used

The Python implementation uses a synthetic customer dataset containing:

### Numerical Features

- Age
- Income
- Tenure
- MonthlyCharges

### Categorical Features

- Gender
- Contract
- City

### Target

- Churn

The dataset is intentionally designed to demonstrate feature creation, transformations, encoding, feature selection, and class imbalance.

---

# 🎯 Key Takeaway

Feature engineering is not about blindly creating more columns.

The objective is:

```text
Raw Information
      ↓
Meaningful Representation
      ↓
Useful Features
      ↓
Better Learning
```

More features do not automatically mean a better model.

Good feature engineering should improve the information available to the model while avoiding leakage, unnecessary complexity, and noise.
