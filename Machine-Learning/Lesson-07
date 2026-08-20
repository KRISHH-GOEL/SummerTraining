# Lesson 07 – Model Evaluation, Validation & Hyperparameter Tuning

## 📌 Objective

The objective of this lesson was to understand how Machine Learning models should be properly evaluated, validated, compared, and improved.

Training a model and obtaining a high training score does not necessarily mean that the model will perform well on unseen data.

This lesson focuses on building reliable evaluation workflows.

---

# 📚 Topics Covered

## Model Evaluation

- Training performance
- Validation performance
- Testing performance
- Generalization
- Underfitting
- Overfitting
- Bias
- Variance

## Cross-Validation

- Why cross-validation is required
- K-Fold Cross-Validation
- Stratified K-Fold
- Cross-validation scores
- Mean CV score
- Standard deviation of CV score
- `cross_val_score()`
- `cross_validate()`

## Hyperparameters

- Parameters vs hyperparameters
- Why hyperparameter tuning is required
- Manual tuning
- Grid Search
- Random Search

## Hyperparameter Tuning

- `GridSearchCV`
- `RandomizedSearchCV`
- Parameter grids
- Parameter distributions
- Best parameters
- Best estimator
- Best cross-validation score

## Model Selection

- Comparing multiple models
- Validation-based model selection
- Cross-validation based selection
- Avoiding test-set leakage

## Classification Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Regression Evaluation

- MAE
- MSE
- RMSE
- R²
- Adjusted R² concept

## Learning Curves

- Training score
- Validation score
- Detecting overfitting
- Detecting underfitting

## Pipelines

- Why pipelines are useful
- Combining preprocessing and model training
- Preventing data leakage
- Using pipelines during cross-validation and tuning

---

# 🧠 Parameters vs Hyperparameters

A model learns its parameters during training.

Example:

```text
Linear Regression
→ coefficients

Decision Tree
→ split rules
```

Hyperparameters are selected before training.

Examples:

```text
Decision Tree
→ max_depth
→ min_samples_split

Random Forest
→ n_estimators
→ max_depth

KNN
→ n_neighbors

Logistic Regression
→ C
```

---

# ⚠️ Why Training Accuracy Is Not Enough

Suppose:

```text
Training Accuracy = 99%
Testing Accuracy  = 75%
```

The model may have memorized patterns specific to the training data.

This is a common sign of:

```text
Overfitting
```

A better model should generalize to unseen data.

---

# 🔴 Underfitting

Underfitting occurs when a model is too simple to capture important patterns.

Example:

```text
Training Score = 65%
Testing Score  = 63%
```

Both performances are poor.

Possible solutions:

- Increase model complexity
- Add useful features
- Reduce excessive regularization
- Use a more appropriate algorithm

---

# 🔴 Overfitting

Overfitting occurs when a model performs very well on training data but poorly on unseen data.

Example:

```text
Training Score = 99%
Testing Score  = 72%
```

Possible solutions:

- Reduce model complexity
- Regularization
- More training data
- Feature selection
- Cross-validation
- Early stopping where applicable
- Hyperparameter tuning

---

# 🔄 Cross-Validation

Instead of relying on one train-validation split, K-Fold Cross-Validation divides the training data into multiple folds.

Example:

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training

Fold 2 → Validation
Fold 1 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training

...
```

The process continues until every fold has been used as validation data.

---

# 📊 K-Fold Cross-Validation

For:

```text
K = 5
```

the dataset is divided into five folds.

Five validation scores are produced:

```text
Score 1
Score 2
Score 3
Score 4
Score 5
```

The final CV score is commonly summarized using:

```text
Mean Score
```

The standard deviation provides information about variability across folds.

---

# 🧮 Stratified K-Fold

For classification problems, preserving class proportions across folds is often important.

Example:

```text
Original:

Class 0 → 80%
Class 1 → 20%
```

Each fold attempts to maintain a similar class distribution.

This is why:

```python
StratifiedKFold
```

is commonly used for classification.

---

# 🔍 Grid Search

Grid Search evaluates combinations from a predefined hyperparameter grid.

Example:

```python
param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10]
}
```

The algorithm evaluates combinations such as:

```text
3, 2
3, 5
3, 10
5, 2
5, 5
5, 10
7, 2
7, 5
7, 10
```

The best combination is selected according to the chosen scoring metric.

---

# 🎲 Randomized Search

Randomized Search samples combinations from specified parameter distributions.

It can be considerably more efficient when the search space is large.

Example:

```python
RandomizedSearchCV(
    model,
    param_distributions=...,
    n_iter=20
)
```

Instead of evaluating every possible combination, it evaluates a selected number of combinations.

---

# 🆚 Grid Search vs Random Search

| Grid Search | Random Search |
|---|---|
| Tests specified combinations | Samples combinations |
| Can become expensive | Often more computationally efficient |
| Exhaustive over provided grid | Does not test every combination |
| Useful for smaller search spaces | Useful for larger search spaces |

---

# 🎯 Choosing the Right Scoring Metric

The best metric depends on the problem.

## Classification

Possible metrics:

```text
accuracy
precision
recall
f1
roc_auc
average_precision
```

For an imbalanced disease detection problem, for example, recall may be more important than accuracy.

---

## Regression

Possible metrics:

```text
neg_mean_absolute_error
neg_mean_squared_error
neg_root_mean_squared_error
r2
```

---

# 📏 Regression Metrics

## MAE

Mean Absolute Error:

```text
MAE = average(|actual - predicted|)
```

It represents the average absolute prediction error.

---

## MSE

Mean Squared Error:

```text
MSE = average((actual - predicted)²)
```

Large errors receive more penalty.

---

## RMSE

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target variable.

---

## R²

R² measures the proportion of variance explained by the model.

Higher is generally better.

---

# 📈 Learning Curves

Learning curves show how model performance changes as the training dataset size increases.

They can help identify:

- Underfitting
- Overfitting
- Whether more training data might help

Conceptually:

```text
Performance
    │
    │      Validation
    │     ────────────
    │    /
    │   /
    │  /
    │ /
    │────────────── Training
    └──────────────────────
          Training Size
```

The actual pattern depends on the model and dataset.

---

# 🔐 Data Leakage

Data leakage occurs when information that should not be available during training is used in a way that gives the model unfair information.

Examples:

- Scaling before train-test splitting
- Imputing using the complete dataset
- Using test information during feature engineering
- Selecting features using the entire dataset before CV
- Tuning directly against the test set

A proper pipeline helps prevent many preprocessing-related leakage problems.

---

# 🔄 Recommended ML Evaluation Workflow

```text
Dataset
   ↓
Train-Test Split
   ↓
Keep Test Set Untouched
   ↓
Build Pipeline
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Select Best Model
   ↓
Evaluate Once on Test Set
   ↓
Final Performance
```

The test set should ideally be used only for final unbiased evaluation after model development decisions are complete.

---

# 🧪 Dataset Used

The Python implementation uses the Breast Cancer dataset available through Scikit-learn.

It is a binary classification problem:

```text
0 → Malignant
1 → Benign
```

The implementation compares multiple classifiers and tunes a Random Forest model.

---

# 📝 Mini Practice

1. Perform 5-fold cross-validation.

2. Compare K-Fold and Stratified K-Fold.

3. Calculate mean and standard deviation of CV scores.

4. Train Logistic Regression.

5. Train Decision Tree.

6. Train Random Forest.

7. Compare their cross-validation scores.

8. Perform GridSearchCV on Random Forest.

9. Tune:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

10. Perform RandomizedSearchCV.

11. Compare Grid Search and Random Search.

12. Try different scoring metrics.

13. Plot a learning curve.

14. Identify signs of overfitting.

15. Evaluate the final selected model on the untouched test set.

---

# 🎯 Key Takeaway

A strong Machine Learning workflow is not:

```text
Train Model
↓
Check Accuracy
↓
Done
```

Instead:

```text
Train
↓
Cross-Validate
↓
Tune
↓
Compare
↓
Select
↓
Evaluate on Unseen Test Data
```

The goal is not to build a model that performs perfectly on the training dataset.

The goal is to build a model that **generalizes well to unseen data**.
