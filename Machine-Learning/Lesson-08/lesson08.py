"""
Lesson 08 : Ensemble Learning

Topics Covered
---------------
1. Ensemble Learning
2. Voting
3. Hard Voting
4. Soft Voting
5. Bagging
6. Random Forest
7. Feature Importance
8. AdaBoost
9. Gradient Boosting
10. Stacking
11. Cross-Validation
12. Model Comparison
13. Final Test Evaluation

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

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("ENSEMBLE LEARNING")
print("=" * 80)

print("""
Ensemble Learning combines multiple Machine Learning
models to produce a stronger final prediction.

The main techniques covered in this lesson are:

- Voting
- Bagging
- Random Forest
- AdaBoost
- Gradient Boosting
- Stacking
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

print("\nShape:")

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
# BASE MODELS
# ==========================================================

print("\n" + "=" * 80)
print("CREATING BASE MODELS")
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


decision_tree = DecisionTreeClassifier(

    max_depth=5,

    random_state=42

)


knn_model = Pipeline(

    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            KNeighborsClassifier(
                n_neighbors=7
            )
        )

    ]

)


# ==========================================================
# RANDOM FOREST
# ==========================================================

random_forest = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    min_samples_split=5,

    random_state=42,

    n_jobs=-1

)


# ==========================================================
# BAGGING
# ==========================================================

bagging_model = BaggingClassifier(

    estimator=DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    n_estimators=100,

    max_samples=0.8,

    max_features=0.8,

    bootstrap=True,

    random_state=42,

    n_jobs=-1

)


# ==========================================================
# ADABOOST
# ==========================================================

adaboost_model = AdaBoostClassifier(

    n_estimators=100,

    learning_rate=0.5,

    random_state=42

)


# ==========================================================
# GRADIENT BOOSTING
# ==========================================================

gradient_boosting_model = (
    GradientBoostingClassifier(

        n_estimators=150,

        learning_rate=0.05,

        max_depth=3,

        random_state=42

    )
)


# ==========================================================
# HARD VOTING
# ==========================================================

print("\n" + "=" * 80)
print("HARD VOTING")
print("=" * 80)

hard_voting = VotingClassifier(

    estimators=[

        (
            "logistic",
            logistic_model
        ),

        (
            "tree",
            decision_tree
        ),

        (
            "knn",
            knn_model
        )

    ],

    voting="hard"

)


# ==========================================================
# SOFT VOTING
# ==========================================================

print("\n" + "=" * 80)
print("SOFT VOTING")
print("=" * 80)

soft_voting = VotingClassifier(

    estimators=[

        (
            "logistic",
            logistic_model
        ),

        (
            "tree",
            decision_tree
        ),

        (
            "knn",
            knn_model
        )

    ],

    voting="soft"

)


# ==========================================================
# STACKING
# ==========================================================

print("\n" + "=" * 80)
print("STACKING")
print("=" * 80)

stacking_model = StackingClassifier(

    estimators=[

        (
            "logistic",
            logistic_model
        ),

        (
            "tree",
            decision_tree
        ),

        (
            "knn",
            knn_model
        )

    ],

    final_estimator=LogisticRegression(
        max_iter=5000
    ),

    cv=5,

    n_jobs=-1

)


# ==========================================================
# MODEL COLLECTION
# ==========================================================

models = {

    "Logistic Regression":
        logistic_model,

    "Decision Tree":
        decision_tree,

    "KNN":
        knn_model,

    "Random Forest":
        random_forest,

    "Bagging":
        bagging_model,

    "AdaBoost":
        adaboost_model,

    "Gradient Boosting":
        gradient_boosting_model,

    "Hard Voting":
        hard_voting,

    "Soft Voting":
        soft_voting,

    "Stacking":
        stacking_model

}


# ==========================================================
# CROSS-VALIDATION
# ==========================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


scoring = {

    "accuracy":
        "accuracy",

    "precision":
        "precision",

    "recall":
        "recall",

    "f1":
        "f1",

    "roc_auc":
        "roc_auc"

}


results = []


for name, model in models.items():

    print(
        f"\nEvaluating: {name}"
    )

    cv_result = cross_validate(

        model,

        X_train,

        y_train,

        cv=skf,

        scoring=scoring,

        n_jobs=-1

    )

    results.append({

        "Model":
            name,

        "Accuracy":
            cv_result[
                "test_accuracy"
            ].mean(),

        "Precision":
            cv_result[
                "test_precision"
            ].mean(),

        "Recall":
            cv_result[
                "test_recall"
            ].mean(),

        "F1":
            cv_result[
                "test_f1"
            ].mean(),

        "ROC-AUC":
            cv_result[
                "test_roc_auc"
            ].mean()

    })


# ==========================================================
# MODEL COMPARISON
# ==========================================================

comparison_df = pd.DataFrame(
    results
)

comparison_df = (
    comparison_df
    .sort_values(
        "F1",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("ENSEMBLE MODEL COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)


# ==========================================================
# F1 SCORE VISUALIZATION
# ==========================================================

plt.figure(
    figsize=(12, 7)
)

plt.barh(

    comparison_df["Model"],

    comparison_df["F1"]

)

plt.xlabel(
    "Mean Cross-Validation F1 Score"
)

plt.ylabel(
    "Model"
)

plt.title(
    "Model Comparison using F1 Score"
)

plt.gca().invert_yaxis()

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "ensemble_model_comparison.png",
    dpi=300
)

plt.show()


# ==========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

random_forest.fit(
    X_train,
    y_train
)

feature_importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        random_forest.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    feature_importance
    .head(15)
    .to_string(
        index=False
    )
)


# ==========================================================
# FEATURE IMPORTANCE VISUALIZATION
# ==========================================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)

plt.xlabel(
    "Feature Importance"
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
# TRAIN FINAL ENSEMBLE
# ==========================================================

print("\n" + "=" * 80)
print("FINAL ENSEMBLE EVALUATION")
print("=" * 80)

# Select the ensemble that performed strongly
# during cross-validation.
#
# Soft Voting is selected here as an example.
# In a real project, selection should be based
# on the complete validation results and business goal.

final_model = soft_voting

final_model.fit(
    X_train,
    y_train
)


# ==========================================================
# FINAL PREDICTIONS
# ==========================================================

final_predictions = (
    final_model.predict(
        X_test
    )
)

final_probabilities = (
    final_model.predict_proba(
        X_test
    )[:, 1]
)


# ==========================================================
# FINAL METRICS
# ==========================================================

final_accuracy = accuracy_score(

    y_test,

    final_predictions

)

final_precision = precision_score(

    y_test,

    final_predictions

)

final_recall = recall_score(

    y_test,

    final_predictions

)

final_f1 = f1_score(

    y_test,

    final_predictions

)

final_roc_auc = roc_auc_score(

    y_test,

    final_probabilities

)


print(
    f"Accuracy  : {final_accuracy:.4f}"
)

print(
    f"Precision : {final_precision:.4f}"
)

print(
    f"Recall    : {final_recall:.4f}"
)

print(
    f"F1 Score  : {final_f1:.4f}"
)

print(
    f"ROC-AUC   : {final_roc_auc:.4f}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(

    classification_report(

        y_test,

        final_predictions,

        target_names=[

            "Malignant",

            "Benign"

        ]

    )

)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_test,

    final_predictions

)

print("\nConfusion Matrix:")

print(cm)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[

        "Malignant",

        "Benign"

    ]

)

display.plot()

plt.title(
    "Soft Voting Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "soft_voting_confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================================
# HARD VS SOFT VOTING
# ==========================================================

print("\n" + "=" * 80)
print("HARD VOTING VS SOFT VOTING")
print("=" * 80)


hard_voting.fit(
    X_train,
    y_train
)

soft_voting.fit(
    X_train,
    y_train
)


hard_predictions = (
    hard_voting.predict(
        X_test
    )
)

soft_predictions = (
    soft_voting.predict(
        X_test
    )
)


hard_f1 = f1_score(

    y_test,

    hard_predictions

)

soft_f1 = f1_score(

    y_test,

    soft_predictions

)


print(
    "Hard Voting F1:",
    round(
        hard_f1,
        4
    )
)

print(
    "Soft Voting F1:",
    round(
        soft_f1,
        4
    )
)


# ==========================================================
# BAGGING EXPLANATION
# ==========================================================

print("\n" + "=" * 80)
print("BAGGING")
print("=" * 80)

print("""
Bagging trains multiple models using bootstrap samples.

The models are trained independently.

This primarily helps reduce variance.

Random Forest is an advanced bagging-style ensemble
that introduces additional randomness through feature
selection.
""")


# ==========================================================
# BOOSTING EXPLANATION
# ==========================================================

print("\n" + "=" * 80)
print("BOOSTING")
print("=" * 80)

print("""
Boosting trains weak learners sequentially.

Each subsequent learner attempts to improve upon the
errors made by the previous learners.

Examples:

- AdaBoost
- Gradient Boosting

Modern algorithms such as XGBoost, LightGBM and CatBoost
extend the boosting family with additional optimizations
and capabilities.
""")


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Ensemble Learning
✔ Weak Learners
✔ Voting
✔ Hard Voting
✔ Soft Voting
✔ Bagging
✔ Random Forest
✔ Feature Importance
✔ AdaBoost
✔ Gradient Boosting
✔ Stacking
✔ Cross-Validation
✔ Model Comparison
✔ Classification Metrics
✔ Ensemble Evaluation
""")

print("\nLesson 08 Completed Successfully!")
