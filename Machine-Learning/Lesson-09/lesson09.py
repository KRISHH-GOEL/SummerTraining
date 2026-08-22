"""
Lesson 09 : Model Explainability & Feature Importance

Topics Covered
---------------
1. Model Interpretability
2. Global Explainability
3. Local Explainability
4. Logistic Regression Coefficients
5. Random Forest Feature Importance
6. Permutation Importance
7. Partial Dependence
8. SHAP Concept
9. Feature Contribution
10. Correlation vs Causation

Author : Krish Goel
Repository : Summer Training
"""


# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import (
    permutation_importance,
    PartialDependenceDisplay
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("MODEL EXPLAINABILITY & FEATURE IMPORTANCE")
print("=" * 80)

print("""
Model explainability helps us understand how Machine
Learning models use input features to make predictions.

This lesson explores:

- Model coefficients
- Tree feature importance
- Permutation importance
- Partial dependence
- SHAP concepts
""")


# ==========================================================
# LOAD DATASET
# ==========================================================

print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

data = load_breast_cancer()

X = pd.DataFrame(

    data.data,

    columns=data.feature_names

)

y = pd.Series(

    data.target,

    name="Target"

)


print("\nDataset Shape:")

print(X.shape)

print("\nTarget Distribution:")

print(
    y.value_counts()
    .sort_index()
)


# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )
)


print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION")
print("=" * 80)


logistic_model = Pipeline(

    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=5000
            )
        )

    ]

)


logistic_model.fit(

    X_train,

    y_train

)


logistic_predictions = (
    logistic_model.predict(
        X_test
    )
)


logistic_probabilities = (
    logistic_model.predict_proba(
        X_test
    )[:, 1]
)


print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            logistic_predictions
        ),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            logistic_predictions
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            logistic_probabilities
        ),
        4
    )
)


# ==========================================================
# LOGISTIC REGRESSION COEFFICIENTS
# ==========================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 80)


logistic_coefficients = (
    logistic_model
    .named_steps["model"]
    .coef_[0]
)


coefficient_df = pd.DataFrame({

    "Feature":
        X.columns,

    "Coefficient":
        logistic_coefficients,

    "Absolute_Coefficient":
        np.abs(
            logistic_coefficients
        )

})


coefficient_df = (
    coefficient_df
    .sort_values(
        "Absolute_Coefficient",
        ascending=False
    )
)


print(
    coefficient_df
    .head(15)
    .to_string(
        index=False
    )
)


# ==========================================================
# COEFFICIENT VISUALIZATION
# ==========================================================

top_coefficients = (
    coefficient_df
    .head(15)
    .sort_values(
        "Coefficient"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(

    top_coefficients["Feature"],

    top_coefficients["Coefficient"]

)

plt.xlabel(
    "Coefficient"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Logistic Regression Coefficients"
)

plt.tight_layout()

plt.savefig(
    "logistic_coefficients.png",
    dpi=300
)

plt.show()


# ==========================================================
# INTERPRETING COEFFICIENT DIRECTION
# ==========================================================

print("\n" + "=" * 80)
print("COEFFICIENT INTERPRETATION")
print("=" * 80)

print("""
Positive coefficient:
    Increasing the feature increases the model's
    linear decision function, holding other model
    inputs constant.

Negative coefficient:
    Increasing the feature decreases the model's
    linear decision function, holding other model
    inputs constant.

Important:
A coefficient describes a predictive relationship
within the fitted model. It does NOT establish causation.
""")


# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM FOREST")
print("=" * 80)


random_forest = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    min_samples_split=5,

    random_state=42,

    n_jobs=-1

)


random_forest.fit(

    X_train,

    y_train

)


rf_predictions = (
    random_forest.predict(
        X_test
    )
)


rf_probabilities = (
    random_forest.predict_proba(
        X_test
    )[:, 1]
)


print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            rf_predictions
        ),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            rf_predictions
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            rf_probabilities
        ),
        4
    )
)


# ==========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)


rf_importance_df = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        random_forest.feature_importances_

})


rf_importance_df = (
    rf_importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    rf_importance_df
    .head(15)
    .to_string(
        index=False
    )
)


# ==========================================================
# RANDOM FOREST IMPORTANCE VISUALIZATION
# ==========================================================

top_rf_features = (
    rf_importance_df
    .head(15)
    .sort_values(
        "Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(

    top_rf_features["Feature"],

    top_rf_features["Importance"]

)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "random_forest_feature_importance.png",
    dpi=300
)

plt.show()


# ==========================================================
# PERMUTATION IMPORTANCE
# ==========================================================

print("\n" + "=" * 80)
print("PERMUTATION IMPORTANCE")
print("=" * 80)


permutation_result = permutation_importance(

    random_forest,

    X_test,

    y_test,

    scoring="f1",

    n_repeats=10,

    random_state=42,

    n_jobs=-1

)


permutation_df = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance_Mean":
        permutation_result.importances_mean,

    "Importance_Std":
        permutation_result.importances_std

})


permutation_df = (
    permutation_df
    .sort_values(
        "Importance_Mean",
        ascending=False
    )
)


print(
    permutation_df
    .head(15)
    .to_string(
        index=False
    )
)


# ==========================================================
# PERMUTATION IMPORTANCE VISUALIZATION
# ==========================================================

top_permutation = (
    permutation_df
    .head(15)
    .sort_values(
        "Importance_Mean"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(

    top_permutation["Feature"],

    top_permutation["Importance_Mean"],

    xerr=top_permutation[
        "Importance_Std"
    ]

)

plt.xlabel(
    "Mean F1 Score Decrease"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Permutation Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "permutation_importance.png",
    dpi=300
)

plt.show()


# ==========================================================
# COMPARE IMPORTANCE METHODS
# ==========================================================

print("\n" + "=" * 80)
print("COMPARING FEATURE IMPORTANCE METHODS")
print("=" * 80)


importance_comparison = (
    rf_importance_df
    .merge(

        permutation_df,

        on="Feature"

    )
)


importance_comparison = (
    importance_comparison
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    importance_comparison
    .head(15)
    .to_string(
        index=False
    )
)


# ==========================================================
# PARTIAL DEPENDENCE
# ==========================================================

print("\n" + "=" * 80)
print("PARTIAL DEPENDENCE")
print("=" * 80)


# Select two important features for demonstration.

important_features = (
    rf_importance_df
    .head(2)
    ["Feature"]
    .tolist()
)


print(
    "Selected features:",
    important_features
)


PartialDependenceDisplay.from_estimator(

    random_forest,

    X_test,

    important_features,

    kind="average"

)

plt.suptitle(
    "Partial Dependence"
)

plt.tight_layout()

plt.savefig(
    "partial_dependence.png",
    dpi=300
)

plt.show()


# ==========================================================
# LOCAL PREDICTION EXAMPLE
# ==========================================================

print("\n" + "=" * 80)
print("LOCAL PREDICTION EXAMPLE")
print("=" * 80)


sample_index = 0

sample = (
    X_test
    .iloc[
        [sample_index]
    ]
)


actual_value = (
    y_test
    .iloc[
        sample_index
    ]
)


predicted_value = (
    random_forest
    .predict(
        sample
    )[0]
)


predicted_probability = (
    random_forest
    .predict_proba(
        sample
    )[0, 1]
)


print(
    "Actual Class:",
    actual_value
)

print(
    "Predicted Class:",
    predicted_value
)

print(
    "Predicted Probability:",
    round(
        predicted_probability,
        4
    )
)


print("\nInput Features:")

print(
    sample.T.to_string(
        header=False
    )
)


# ==========================================================
# SHAP CONCEPT
# ==========================================================

print("\n" + "=" * 80)
print("SHAP CONCEPT")
print("=" * 80)

print("""
SHAP attempts to explain individual predictions by
assigning contribution values to features.

Conceptually:

Base Prediction
       +
Feature A Contribution
       +
Feature B Contribution
       +
Feature C Contribution
       =
Final Prediction

SHAP can provide both:

1. Local explanations
   → Why was this particular observation predicted this way?

2. Global explanations
   → Which features generally contribute most across observations?

The SHAP package is not required for the rest of this
lesson. It can be installed separately when you are ready
to experiment with SHAP directly.

Installation:

pip install shap
""")


# ==========================================================
# OPTIONAL SHAP EXAMPLE
# ==========================================================

print("\n" + "=" * 80)
print("OPTIONAL SHAP IMPLEMENTATION")
print("=" * 80)

print("""
Example:

import shap

explainer = shap.TreeExplainer(random_forest)

shap_values = explainer.shap_values(
    X_test
)

shap.summary_plot(
    shap_values,
    X_test
)

For binary classification, inspect the SHAP package's
current output structure because it can vary across
versions and explainers.
""")


# ==========================================================
# IMPORTANT LIMITATIONS
# ==========================================================

print("\n" + "=" * 80)
print("IMPORTANT LIMITATIONS")
print("=" * 80)

print("""
Feature importance does not mean causation.

A feature may appear important because:

- It contains useful predictive information.
- It is correlated with another important feature.
- It acts as a proxy for another variable.
- The model has learned a dataset-specific relationship.

Therefore:

Predictive Importance
        ≠
Causal Importance
""")


# ==========================================================
# FINAL COMPARISON
# ==========================================================

print("\n" + "=" * 80)
print("EXPLAINABILITY METHODS")
print("=" * 80)

explainability_methods = pd.DataFrame({

    "Method": [

        "Logistic Regression Coefficients",

        "Random Forest Feature Importance",

        "Permutation Importance",

        "Partial Dependence",

        "SHAP"

    ],

    "Main_Purpose": [

        "Understand linear model coefficients",

        "Understand tree split-based importance",

        "Measure performance drop after shuffling",

        "Study average feature-response relationship",

        "Explain global and individual predictions"

    ]

})


print(
    explainability_methods.to_string(
        index=False
    )
)


# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Change the Random Forest hyperparameters.

2. Compare feature importance before and after tuning.

3. Compare:

   Logistic Regression coefficients
   Random Forest importance
   Permutation importance

4. Investigate why the rankings differ.

5. Change the permutation scoring metric.

6. Increase n_repeats.

7. Generate Partial Dependence for different features.

8. Install SHAP.

9. Generate a SHAP summary plot.

10. Generate a local explanation for one observation.

11. Investigate highly correlated features.

12. Explain why importance does not imply causation.
""")


# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Model Explainability
✔ Global Interpretability
✔ Local Interpretability
✔ Logistic Regression Coefficients
✔ Random Forest Feature Importance
✔ Permutation Importance
✔ Partial Dependence
✔ SHAP Concepts
✔ Feature Contributions
✔ Model Interpretation
✔ Correlation vs Causation
✔ Responsible Interpretation
""")

print("\nLesson 09 Completed Successfully!")
