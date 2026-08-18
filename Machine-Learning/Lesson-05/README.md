# Lesson 05 – Classification Algorithms

## 📌 Objective

The objective of this lesson was to understand Classification as a supervised Machine Learning problem and implement major classification algorithms using Scikit-learn.

Classification is used when the target variable represents a category or class.

Examples:

- Spam vs Not Spam
- Disease vs No Disease
- Churn vs No Churn
- Fraud vs Legitimate
- Customer Segment
- Image Category

---

# 📚 Topics Covered

## Classification Fundamentals

- What is Classification?
- Binary Classification
- Multiclass Classification
- Multilabel Classification
- Features and target
- Decision boundary
- Class probabilities

## Logistic Regression

- Logistic Regression
- Sigmoid function
- Probability prediction
- Decision threshold
- `LogisticRegression()`

## K-Nearest Neighbors

- KNN concept
- Distance-based classification
- Choosing `k`
- `KNeighborsClassifier()`
- Importance of feature scaling

## Decision Tree Classification

- Decision rules
- Splitting
- Gini impurity
- Entropy
- Information gain
- Tree depth
- Overfitting

## Random Forest Classification

- Ensemble learning
- Multiple decision trees
- Bootstrap sampling
- Random feature selection
- Majority voting
- `RandomForestClassifier()`

## Support Vector Machine

- Hyperplane
- Maximum margin
- Support vectors
- Kernel concept
- `SVC()`

## Naive Bayes

- Bayes theorem
- Conditional probability
- Naive independence assumption
- Gaussian Naive Bayes
- `GaussianNB()`

---

# 📊 Classification Types

## Binary Classification

Only two classes exist.

Example:

```text
0 → No Churn
1 → Churn
```

---

## Multiclass Classification

More than two mutually exclusive classes.

Example:

```text
0 → Cat
1 → Dog
2 → Bird
```

---

## Multilabel Classification

One observation can belong to multiple classes simultaneously.

Example:

```text
Movie:
Action = 1
Comedy = 1
Drama = 0
```

---

# 🧠 Logistic Regression

Despite its name, Logistic Regression is primarily used for classification.

It calculates a probability using the sigmoid function:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

The output lies between:

```text
0 and 1
```

Example:

```text
P(Churn) = 0.82
```

A threshold can then be used to convert the probability into a class prediction.

Default binary threshold:

```text
Probability >= 0.5 → Class 1
Probability < 0.5  → Class 0
```

The threshold can be changed depending on the business objective.

---

# 📍 K-Nearest Neighbors

KNN predicts a class based on nearby observations.

General workflow:

```text
New Observation
       ↓
Calculate Distance
       ↓
Find K Nearest Points
       ↓
Majority Vote
       ↓
Prediction
```

Important parameter:

```text
n_neighbors
```

KNN is sensitive to feature scale because distance calculations are involved.

---

# 🌳 Decision Tree

A Decision Tree learns rules that split the data.

Example:

```text
Age > 40?
    │
 ┌──┴──┐
Yes    No
 │      │
Income > 50K?
```

Important parameters:

```text
max_depth
min_samples_split
min_samples_leaf
criterion
```

Common split criteria:

```text
gini
entropy
log_loss
```

Decision Trees do not require feature scaling.

---

# 🌲 Random Forest

Random Forest is an ensemble of Decision Trees.

```text
Dataset
   ↓
Tree 1 ──→ Prediction
Tree 2 ──→ Prediction
Tree 3 ──→ Prediction
Tree 4 ──→ Prediction
...
   ↓
Majority Voting
   ↓
Final Class
```

Important parameters:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

---

# ⚙️ Support Vector Machine

SVM attempts to find a decision boundary that separates classes while maximizing the margin.

Important concepts:

- Hyperplane
- Margin
- Support vectors
- Kernel
- C
- Gamma

Common kernels:

```text
linear
rbf
poly
sigmoid
```

SVM generally benefits from feature scaling.

---

# 🧮 Naive Bayes

Naive Bayes is based on Bayes theorem:

```text
P(A|B) = P(B|A)P(A) / P(B)
```

The "naive" assumption is that features are conditionally independent given the class.

For continuous numerical features, one common implementation is:

```python
GaussianNB()
```

Naive Bayes is computationally efficient and can work particularly well for some high-dimensional problems.

---

# 📏 Classification Evaluation Metrics

Accuracy alone is not always sufficient.

## Accuracy

```text
Correct Predictions
-------------------
Total Predictions
```

Useful when classes are reasonably balanced and the costs of errors are similar.

---

## Confusion Matrix

A binary confusion matrix contains:

```text
                 Predicted
                0       1

Actual  0      TN      FP

        1      FN      TP
```

Where:

- TP = True Positive
- TN = True Negative
- FP = False Positive
- FN = False Negative

---

## Precision

```text
TP
---------
TP + FP
```

Of all predicted positives, how many were actually positive?

---

## Recall

```text
TP
---------
TP + FN
```

Of all actual positives, how many were correctly identified?

Recall is especially important when missing a positive case is costly.

---

## F1 Score

Harmonic mean of precision and recall:

```text
F1 = 2 × Precision × Recall
          ------------------
          Precision + Recall
```

---

## ROC-AUC

Measures ranking/discrimination ability across classification thresholds.

A higher ROC-AUC generally indicates better separation between classes.

---

## Precision-Recall AUC

PR-AUC can be particularly informative when the positive class is relatively rare.

---

# ⚠️ Class Imbalance

Example:

```text
Class 0 → 950
Class 1 → 50
```

A model predicting everything as Class 0 gets:

```text
95% accuracy
```

but completely fails to identify the minority class.

Therefore, when classes are imbalanced, inspect:

- Precision
- Recall
- F1
- Confusion Matrix
- ROC-AUC
- PR-AUC

---

# 🎚️ Decision Threshold

A classifier may output:

```text
P(Class 1)
```

Example:

```text
0.91
0.76
0.54
0.31
0.12
```

Changing the threshold changes the classification decision.

For example:

```text
Threshold = 0.50
```

versus:

```text
Threshold = 0.30
```

A lower threshold usually increases recall but can also increase false positives.

The correct threshold depends on the application.

---

# 🔄 Classification Workflow

```text
Dataset
   ↓
EDA
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
Train Classification Models
   ↓
Predictions
   ↓
Probability Predictions
   ↓
Confusion Matrix
   ↓
Precision / Recall / F1
   ↓
ROC-AUC / PR-AUC
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
```

---

# 📦 Libraries Used

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# 🧪 Dataset Used

The Python implementation uses a synthetic customer dataset containing:

### Features

- Age
- Income
- Tenure
- MonthlyCharges
- SupportCalls

### Target

```text
Churn
```

Where:

```text
0 → Customer did not churn
1 → Customer churned
```

The dataset is intentionally generated with a meaningful relationship between features and churn so that the algorithms can be compared.

---

# 🎯 Key Takeaway

Different classification algorithms make different assumptions.

```text
Logistic Regression
→ Linear decision boundary + probabilities

KNN
→ Distance and neighboring observations

Decision Tree
→ Rule-based splits

Random Forest
→ Ensemble of decision trees

SVM
→ Maximum-margin decision boundary

Naive Bayes
→ Probabilistic model with conditional independence assumption
```

There is no universally best classifier.

The appropriate model should be selected using validation performance, business requirements, computational constraints, and interpretability.

---

# 📝 Mini Practice

1. Train Logistic Regression.

2. Train KNN with different values of `k`.

3. Train a Decision Tree.

4. Change the tree's `max_depth`.

5. Train a Random Forest.

6. Train an SVM using:
   - Linear kernel
   - RBF kernel

7. Train Gaussian Naive Bayes.

8. Compare all classifiers using:
   - Accuracy
   - Precision
   - Recall
   - F1
   - ROC-AUC

9. Generate a confusion matrix for every model.

10. Compare `predict()` and `predict_proba()`.

11. Change the classification threshold.

12. Determine which model is best for detecting churn.

13. Explain why accuracy alone may be misleading for an imbalanced dataset.

14. Compare model training and testing performance to identify overfitting.

15. Experiment with class weights using:

```python
class_weight="balanced"
```
