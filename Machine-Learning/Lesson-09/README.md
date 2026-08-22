# Lesson 09 – Model Explainability & Feature Importance

## 📌 Objective

The objective of this lesson is to understand how Machine Learning models can be interpreted and explained.

A model should not only produce:

    Prediction → 1

It should also help us understand:

    Why → Prediction = 1

Model explainability is especially important when Machine Learning is used in areas such as:

- Healthcare
- Finance
- Banking
- Insurance
- Fraud detection
- Customer analytics
- Risk assessment

---

# 📚 Topics Covered

## Model Interpretability

- What is model interpretability?
- What is model explainability?
- Why explainability matters
- Global vs local explanations
- Black-box models

## Feature Importance

- Feature importance concept
- Tree-based feature importance
- Random Forest feature importance
- Permutation importance

## Model Coefficients

- Linear model coefficients
- Positive coefficients
- Negative coefficients
- Coefficient magnitude

## Permutation Importance

- Feature shuffling
- Performance degradation
- Measuring feature contribution
- `permutation_importance()`

## Partial Dependence

- Partial Dependence Plot
- Feature-response relationship
- `PartialDependenceDisplay`

## SHAP

- SHAP concept
- Shapley values
- Local explanations
- Global explanations
- Feature contribution
- SHAP ecosystem

## Explainability Concepts

- Global interpretability
- Local interpretability
- Feature contribution
- Correlation vs causation
- Model-specific vs model-agnostic methods

---

# 🧠 Why Model Explainability?

Suppose a model predicts:

```text
Patient → High Risk
```

A user may ask:

```text
Why?
```

A useful explanation might indicate:

```text
High blood pressure
High cholesterol
Age
Chest pain
```

The explanation helps humans understand the factors associated with the prediction.

---

# 🌍 Global vs Local Explainability

## Global Explainability

Explains the overall behavior of a model.

Example:

```text
Which features are generally important
across the entire dataset?
```

Methods include:

- Feature importance
- Permutation importance
- Partial dependence

---

## Local Explainability

Explains one particular prediction.

Example:

```text
Why was THIS customer classified as high risk?
```

Methods such as SHAP can provide local feature contributions.

---

# 🌲 Tree-Based Feature Importance

Tree-based models such as:

- Decision Tree
- Random Forest
- Gradient Boosting

can calculate feature importance.

Example:

```text
Feature             Importance

Age                    0.21
Income                 0.18
Feature C              0.15
Feature D              0.10
```

Higher values indicate that the feature contributed more to the model's internal split-based criterion.

However:

> Built-in tree feature importance should not automatically be interpreted as causal importance.

---

# 🔀 Permutation Importance

Permutation importance is model-agnostic.

The basic idea is:

```text
Train Model
    ↓
Measure Baseline Performance
    ↓
Shuffle One Feature
    ↓
Measure Performance Again
    ↓
Calculate Performance Drop
```

If shuffling a feature causes a large performance decrease:

```text
Feature → Important
```

If performance barely changes:

```text
Feature → Less important
```

---

# 📐 Concept

Suppose:

```text
Original F1 = 0.90

Shuffle Age
F1 = 0.82
```

Performance decrease:

```text
0.90 - 0.82 = 0.08
```

Now:

```text
Shuffle Feature B
F1 = 0.895
```

Decrease:

```text
0.90 - 0.895 = 0.005
```

Age appears more important according to this evaluation.

---

# 📊 Model Coefficients

Linear models provide coefficients.

Example:

```text
Prediction =

β₀
+ β₁X₁
+ β₂X₂
+ β₃X₃
```

A positive coefficient indicates that, holding other model inputs constant, increasing the feature tends to increase the model's linear decision function.

A negative coefficient indicates the opposite direction.

The magnitude of coefficients can be compared meaningfully when features are appropriately scaled.

---

# ⚠️ Important

A coefficient does NOT automatically mean:

```text
Feature causes outcome
```

Machine Learning models generally identify predictive relationships.

Therefore:

```text
Association ≠ Causation
```

---

# 📈 Partial Dependence

Partial Dependence helps visualize the model's average predicted response as one or more features vary.

Conceptually:

```text
Feature Value
      ↓
Model Predictions
      ↓
Average Prediction
```

It can help answer questions such as:

```text
How does predicted risk generally change
as this feature increases?
```

---

# 🧩 SHAP

SHAP stands for:

```text
SHapley Additive exPlanations
```

SHAP is based on Shapley values from cooperative game theory.

It attempts to distribute the model's prediction among the input features.

Conceptually:

```text
Base Prediction
      +
Feature A Contribution
      +
Feature B Contribution
      +
Feature C Contribution
      =
Final Prediction
```

---

# 🔬 Local SHAP Explanation

For one observation:

```text
Base value
   +
Age contribution
   +
Income contribution
   +
Feature C contribution
   =
Final prediction
```

This provides an explanation for an individual prediction.

---

# 🌎 Global SHAP Explanation

SHAP values can also be aggregated across observations.

For example:

```text
Feature A → High overall contribution
Feature B → Medium contribution
Feature C → Low contribution
```

This can provide a global view of model behavior.

---

# 🆚 Explainability Techniques

| Technique | Global | Local | Model Agnostic |
|---|---:|---:|---:|
| Tree Feature Importance | ✅ | ❌ | ❌ |
| Coefficients | ✅ | ❌ | ❌ |
| Permutation Importance | ✅ | ❌ | ✅ |
| Partial Dependence | ✅ | Limited | ✅ |
| SHAP | ✅ | ✅ | Mostly* |

`*` SHAP has both model-agnostic and model-specific explainers. The appropriate explainer depends on the model.

---

# ⚠️ Limitations

Explainability methods also have limitations.

### Correlated Features

If two features contain similar information, importance can be distributed between them.

### Feature Importance ≠ Causality

An important predictive feature does not necessarily cause the target.

### Model-Specific Methods

Some methods only work naturally with certain model families.

### Explanations Are Approximations

Many explainability methods summarize model behavior rather than exposing a simple human-readable "reason" inside the model.

---

# 🔄 Explainability Workflow

```text
Dataset
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Global Explanation
   ↓
Feature Importance
   ↓
Permutation Importance
   ↓
Partial Dependence
   ↓
Local Explanation
   ↓
SHAP
   ↓
Business / Domain Interpretation
```

---

# 🧪 Dataset Used

The implementation uses the Breast Cancer dataset available through Scikit-learn.

Models used:

- Logistic Regression
- Random Forest

Explainability techniques demonstrated:

- Logistic Regression coefficients
- Random Forest feature importance
- Permutation importance
- Partial dependence

SHAP is also discussed conceptually and included as an optional section because it requires the external `shap` package.

---

# 📝 Mini Practice

1. Train Logistic Regression.

2. Extract model coefficients.

3. Rank features by absolute coefficient magnitude.

4. Train Random Forest.

5. Extract feature importance.

6. Compare Random Forest importance with Logistic Regression coefficients.

7. Calculate permutation importance.

8. Compare built-in importance with permutation importance.

9. Generate a Partial Dependence Plot.

10. Install SHAP and experiment with a tree model.

11. Generate a local explanation for one observation.

12. Generate a global SHAP summary.

13. Investigate what happens when two features are strongly correlated.

14. Explain why feature importance should not be interpreted as causation.

---

# 🎯 Key Takeaway

A Machine Learning model should ideally answer two questions:

```text
What did the model predict?
```

and

```text
Why did the model predict it?
```

Model explainability helps bridge the gap between:

```text
Machine Learning
       ↓
Human Understanding
       ↓
Trust
       ↓
Business / Real-World Decision Making
```

The goal is not merely to make a model accurate.

The goal is to understand, validate, and responsibly use its predictions.
