# Lesson 02 – Data Preprocessing for Machine Learning

## 📌 Objective

The objective of this lesson was to understand and implement the preprocessing steps required before training Machine Learning models.

Raw datasets commonly contain missing values, categorical variables, different numerical scales, and other inconsistencies.

Machine Learning algorithms generally require data in a suitable numerical and consistent format.

This lesson focuses on building a reliable preprocessing workflow using Scikit-learn.

---

## 📚 Topics Covered

### Feature and Target Separation

- Independent variables `X`
- Dependent variable `y`
- Numerical features
- Categorical features

### Train-Test Split

- Training dataset
- Testing dataset
- `train_test_split()`
- `random_state`
- `test_size`
- Stratified splitting

### Missing Value Handling

- Identifying missing values
- Numerical imputation
- Categorical imputation
- `SimpleImputer`

### Categorical Encoding

- One-Hot Encoding
- `OneHotEncoder`
- `handle_unknown`
- Label Encoding
- When to use each encoding method

### Feature Scaling

- Why scaling is required
- Standardization
- Normalization
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`

### ColumnTransformer

- Applying different transformations to different columns
- Numerical preprocessing
- Categorical preprocessing

### Pipelines

- `Pipeline`
- Combining preprocessing and model training
- Preventing inconsistent transformations

### Data Leakage

- What is data leakage?
- How leakage occurs
- Why preprocessing should be fitted only on training data
- How pipelines help prevent leakage

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Separate features and target variables.
- Split data into training and testing sets.
- Handle missing numerical and categorical values.
- Encode categorical features.
- Scale numerical features.
- Apply different transformations to different columns.
- Build preprocessing pipelines.
- Understand `ColumnTransformer`.
- Identify and avoid data leakage.

---

# 🔄 Standard ML Preprocessing Workflow

```text
Raw Dataset
     ↓
Separate X and y
     ↓
Train-Test Split
     ↓
Identify Numerical & Categorical Features
     ↓
Handle Missing Values
     ↓
Encode Categorical Features
     ↓
Scale Numerical Features
     ↓
Combine Transformations
     ↓
Train Model
```

---

# 🧩 Why Train-Test Split Comes Before Preprocessing

A common mistake is:

```text
Entire Dataset
      ↓
Fit Scaler / Imputer
      ↓
Train-Test Split
```

This can cause **data leakage** because information from the test set can influence preprocessing.

Correct approach:

```text
Raw Dataset
     ↓
Train-Test Split
     ↓
Fit preprocessing on Training Data
     ↓
Transform Training Data
     ↓
Transform Testing Data
```

The test set must remain unseen during training.

---

# 🩹 Missing Value Imputation

### Numerical Features

Common strategies:

```text
Mean
Median
Constant
```

Example:

```python
SimpleImputer(strategy="median")
```

### Categorical Features

Common strategy:

```python
SimpleImputer(strategy="most_frequent")
```

or:

```python
SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
```

---

# 🔤 Categorical Encoding

Machine Learning algorithms generally require numerical inputs.

For example:

```text
Gender

Male
Female
Female
Male
```

can be transformed using One-Hot Encoding:

```text
Gender_Female
Gender_Male
```

### One-Hot Encoding

Best suited for nominal categories where there is no meaningful order.

Example:

```text
Delhi
Mumbai
Chennai
```

### Label Encoding

Maps categories to integer values.

Example:

```text
Low    → 0
Medium → 1
High   → 2
```

Label encoding is particularly appropriate for **ordinal target labels or genuinely ordered categories**. Arbitrarily assigning numbers to nominal feature categories can incorrectly imply an order.

---

# 📏 Feature Scaling

Suppose we have:

```text
Age       → 18–80
Income    → 20,000–2,00,000
```

The numerical ranges are very different.

Some algorithms can be affected by this difference.

---

## Standardization

Transforms data approximately to:

```text
Mean = 0
Standard Deviation = 1
```

Using:

```python
StandardScaler()
```

---

## Min-Max Scaling

Transforms values into a specified range, commonly:

```text
0 to 1
```

Using:

```python
MinMaxScaler()
```

---

## Robust Scaling

Uses the median and interquartile range and is less sensitive to extreme values.

Using:

```python
RobustScaler()
```

---

# 🏗️ ColumnTransformer

Different columns often require different preprocessing.

Example:

```text
Numerical Columns
        ↓
Imputation
        ↓
Scaling

Categorical Columns
        ↓
Imputation
        ↓
One-Hot Encoding
```

`ColumnTransformer` allows these transformations to be applied simultaneously.

---

# 🔗 Pipeline

A Pipeline combines multiple preprocessing steps and a model.

Example:

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])
```

Benefits:

- Consistent preprocessing
- Cleaner code
- Reduced leakage risk
- Easier model deployment
- Reproducibility
- Easier cross-validation and tuning

---

# ⚠️ Data Leakage

Data leakage occurs when information that should be unavailable during training influences the model.

Examples:

- Scaling before train-test split
- Imputing using the complete dataset
- Selecting features using the test set
- Using future information to predict the past
- Including the target or target-derived information as a feature

Leakage can produce unrealistically high evaluation scores.

---

# 📦 Libraries Used

- pandas
- NumPy
- Scikit-learn

Install:

```bash
pip install pandas numpy scikit-learn
```

---

# 🧪 Dataset Used

The Python implementation creates a small mixed-type customer dataset containing:

### Numerical Features

- Age
- Income
- SpendingScore

### Categorical Features

- Gender
- City
- Membership

### Target

- Purchased

The dataset intentionally contains missing values so that the preprocessing pipeline can demonstrate imputation.

---

# 🚀 Complete Preprocessing Pipeline

The implementation follows:

```text
Dataset
   ↓
X / y Separation
   ↓
Train-Test Split
   ↓
Numerical Pipeline
   │
   ├── Median Imputation
   └── Standard Scaling
   │
Categorical Pipeline
   │
   ├── Most-Frequent Imputation
   └── One-Hot Encoding
   │
   ↓
ColumnTransformer
   ↓
Logistic Regression
   ↓
Predictions
```

---

# 📝 Key Takeaway

Preprocessing is not simply about "cleaning the data."

A good ML preprocessing pipeline must ensure:

```text
Correct Transformation
        +
No Data Leakage
        +
Consistent Train/Test Processing
        +
Reproducibility
```

The preprocessing pipeline should eventually become part of the model itself so that the exact same transformations are applied when making predictions on new data.
