# Lesson 08 – Ensemble Learning

## 📌 Objective

The objective of this lesson is to understand Ensemble Learning and how combining multiple Machine Learning models can produce more accurate, stable, and robust predictions.

Instead of relying on a single model, ensemble methods combine the predictions of multiple models.

The major approaches covered in this lesson are:

- Voting
- Bagging
- Random Forest
- Boosting
- AdaBoost
- Gradient Boosting
- XGBoost concept
- Stacking

---

# 📚 Topics Covered

## Ensemble Learning

- What is Ensemble Learning?
- Why combine multiple models?
- Weak learners
- Strong learners
- Diversity among models
- Bias-variance trade-off

## Voting

- Hard Voting
- Soft Voting
- `VotingClassifier`
- Probability-based voting

## Bagging

- Bootstrap Aggregation
- Bootstrap samples
- Parallel model training
- Variance reduction
- `BaggingClassifier`

## Random Forest

- Decision Tree ensemble
- Random feature selection
- Bootstrap sampling
- Multiple decision trees
- Majority voting
- `RandomForestClassifier`
- Feature importance

## Boosting

- Sequential learning
- Weak learners
- Correcting previous errors
- Bias reduction

## AdaBoost

- Adaptive Boosting
- Sample weights
- Misclassified observations
- `AdaBoostClassifier`

## Gradient Boosting

- Gradient Boosting intuition
- Sequential trees
- Residual/error correction
- `GradientBoostingClassifier`

## Stacking

- Base learners
- Meta learner
- Combining different algorithms
- `StackingClassifier`

## Ensemble Evaluation

- Cross-validation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# 🧠 What Is Ensemble Learning?

Ensemble Learning combines predictions from multiple Machine Learning models.

Instead of:

```text
Dataset
   ↓
One Model
   ↓
Prediction
```

we use:

```text
                 ┌── Model 1 ──┐
Dataset ─────────┼── Model 2 ──┼──→ Combined Prediction
                 ├── Model 3 ──┤
                 └── Model 4 ──┘
```

The idea is that several models can collectively make better decisions than a single model.

---

# 🎯 Why Use Ensembles?

A single model may:

- Overfit
- Have high variance
- Have high bias
- Miss certain patterns
- Be sensitive to the training dataset

Combining models can improve:

- Accuracy
- Stability
- Generalization
- Robustness

However, ensemble models are not automatically better.

They still need proper validation and tuning.

---

# 🧩 Weak Learners

A weak learner is a model that performs only slightly better than random guessing.

A collection of weak learners can sometimes be combined into a powerful ensemble.

This is particularly important in:

```text
Boosting
```

---

# 🗳️ Voting

Voting combines predictions from multiple different models.

For example:

```text
Logistic Regression → Class 1
Decision Tree       → Class 1
KNN                 → Class 0

Final → Class 1
```

because the majority voted for Class 1.

---

# 🟢 Hard Voting

Hard voting uses the predicted class labels.

Example:

```text
Model 1 → 1
Model 2 → 1
Model 3 → 0

Final → 1
```

Majority voting determines the final class.

---

# 🔵 Soft Voting

Soft voting uses predicted probabilities.

Example:

```text
Model 1:
Class 0 = 0.20
Class 1 = 0.80

Model 2:
Class 0 = 0.40
Class 1 = 0.60

Model 3:
Class 0 = 0.30
Class 1 = 0.70
```

The probabilities are combined and the class with the highest combined probability is selected.

Soft voting can be useful when the component models provide meaningful probability estimates.

---

# 🎒 Bagging

Bagging stands for:

```text
Bootstrap Aggregating
```

The basic idea is:

```text
Original Dataset
      ↓
Bootstrap Samples
 ┌────┼────┐
 ↓    ↓    ↓
Model Model Model
 ↓    ↓    ↓
 └────┼────┘
      ↓
   Aggregate
      ↓
 Final Prediction
```

Each model receives a different bootstrap sample.

The models can then be trained independently, making bagging naturally parallelizable.

---

# 🌲 Random Forest

Random Forest is one of the most widely used ensemble algorithms.

It combines many Decision Trees.

Random Forest introduces randomness in two important ways:

### 1. Bootstrap Sampling

Different trees are trained using different bootstrap samples.

### 2. Random Feature Selection

Each tree considers a random subset of features when determining splits.

This helps create diverse trees.

---

# 🌳 Random Forest Prediction

For classification:

```text
Tree 1 → Class A
Tree 2 → Class A
Tree 3 → Class B
Tree 4 → Class A
Tree 5 → Class B

Final → Class A
```

The majority vote determines the final class.

---

# ⭐ Feature Importance

Random Forest can estimate feature importance.

This can help identify features that contributed strongly to predictions.

Example:

```text
Feature             Importance

Feature A             0.31
Feature B             0.22
Feature C             0.18
Feature D             0.09
```

However, built-in tree-based feature importance should not automatically be interpreted as causal importance.

---

# 🚀 Boosting

Boosting trains models sequentially.

Instead of training independent models:

```text
Model 1
Model 2
Model 3
```

boosting follows:

```text
Model 1
   ↓
Errors
   ↓
Model 2 focuses on errors
   ↓
Errors
   ↓
Model 3 focuses on remaining errors
```

The final model combines the contributions of the individual learners.

---

# ⚡ AdaBoost

AdaBoost stands for:

```text
Adaptive Boosting
```

The algorithm changes the importance of training observations based on previous errors.

Simplified concept:

```text
Train weak learner
      ↓
Identify mistakes
      ↓
Increase importance of difficult samples
      ↓
Train next learner
      ↓
Repeat
```

---

# 📈 Gradient Boosting

Gradient Boosting builds models sequentially to improve predictions by focusing on the remaining errors.

Conceptually:

```text
Initial Prediction
       ↓
Calculate Error
       ↓
Train Next Tree
       ↓
Improve Prediction
       ↓
Calculate Remaining Error
       ↓
Train Next Tree
       ↓
...
```

Gradient Boosting is a powerful foundation for many modern boosting algorithms.

---

# 🧱 Stacking

Stacking combines different types of models.

Example:

```text
             ┌── Logistic Regression
             │
Dataset ─────┼── Decision Tree
             │
             └── KNN
                    ↓
             Meta Learner
                    ↓
              Final Prediction
```

The first-level models are called:

```text
Base Learners
```

The model that learns how to combine their predictions is called:

```text
Meta Learner
```

---

# 🆚 Bagging vs Boosting

| Bagging | Boosting |
|---|---|
| Models are generally trained independently | Models are trained sequentially |
| Focuses on reducing variance | Often focuses on reducing bias |
| Bootstrap sampling is common | Learners respond to previous errors |
| Easy to parallelize | Sequential dependency |
| Random Forest is a major example | AdaBoost and Gradient Boosting are examples |

---

# 🆚 Major Ensemble Methods

| Method | Main Idea |
|---|---|
| Voting | Combine different model predictions |
| Bagging | Train models on bootstrap samples |
| Random Forest | Bagged randomized decision trees |
| AdaBoost | Focus on incorrectly classified observations |
| Gradient Boosting | Sequentially improve residual errors |
| Stacking | Use a meta-model to combine models |

---

# ⚠️ Ensemble Models Are Not Always Better

More models do not automatically mean better performance.

Potential disadvantages:

- Higher computational cost
- More complexity
- Reduced interpretability
- Longer training time
- Possible overfitting
- More difficult deployment

Therefore, ensembles should be evaluated against strong baseline models.

---

# 🔄 Ensemble Learning Workflow

```text
Dataset
   ↓
Preprocessing
   ↓
Train Baseline Models
   ↓
Evaluate Individual Models
   ↓
Create Ensemble
   ↓
Cross-Validation
   ↓
Tune Hyperparameters
   ↓
Compare Performance
   ↓
Evaluate on Test Set
   ↓
Select Final Model
```

---

# 🧪 Dataset Used

The implementation uses the Breast Cancer dataset available through Scikit-learn.

This is a binary classification problem.

The lesson compares:

- Logistic Regression
- Decision Tree
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- Voting
- Stacking

---

# 📊 Evaluation Metrics

The implementation evaluates models using:

### Accuracy

Overall percentage of correct predictions.

### Precision

Of the observations predicted positive, how many were actually positive?

### Recall

Of the actual positive observations, how many were detected?

### F1 Score

Harmonic mean of precision and recall.

### ROC-AUC

Measures ranking/discrimination ability across classification thresholds.

---

# 📝 Mini Practice

1. Train a Decision Tree baseline.

2. Train a Random Forest.

3. Compare their performance.

4. Create a Bagging Classifier.

5. Try AdaBoost.

6. Try Gradient Boosting.

7. Create a hard voting ensemble.

8. Create a soft voting ensemble.

9. Create a stacking ensemble.

10. Compare all models using cross-validation.

11. Compare accuracy, precision, recall, F1 and ROC-AUC.

12. Inspect Random Forest feature importance.

13. Tune Random Forest.

14. Tune Gradient Boosting.

15. Identify which ensemble provides the best validation performance.

16. Evaluate the final selected model on the untouched test set.

---

# 🎯 Key Takeaway

Ensemble learning is based on a simple but powerful idea:

```text
Don't always depend on one model.
Combine multiple useful models.
```

But the real objective is not:

```text
More models = Better model
```

It is:

```text
Diverse + Strong + Properly Validated Models
                  ↓
          Better Generalization
```

A good Machine Learning practitioner should understand when to use:

```text
Single Model
      ↓
Bagging
      ↓
Random Forest
      ↓
Boosting
      ↓
Voting
      ↓
Stacking
```

and select the approach based on data, performance, computational cost, interpretability, and business requirements.
