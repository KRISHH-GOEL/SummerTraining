"""
Lesson 05 : Classification Algorithms

Topics Covered
---------------
1. Classification Fundamentals
2. Binary Classification
3. Logistic Regression
4. K-Nearest Neighbors
5. Decision Tree Classification
6. Random Forest Classification
7. Support Vector Machine
8. Naive Bayes
9. Confusion Matrix
10. Accuracy
11. Precision
12. Recall
13. F1 Score
14. ROC-AUC
15. PR-AUC
16. Probability Prediction
17. Decision Threshold
18. Model Comparison
19. Overfitting Detection

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("CLASSIFICATION ALGORITHMS")
print("=" * 80)

print("""
Classification is a supervised Machine Learning
problem where the target variable represents
one or more classes.
""")

# ==========================================================
# CREATE DATASET
# ==========================================================

np.random.seed(42)

n_samples = 500

age = np.random.randint(
    18,
    70,
    n_samples
)

income = np.random.randint(
    20000,
    150000,
    n_samples
)

tenure = np.random.randint(
    1,
    72,
    n_samples
)

monthly_charges = np.random.uniform(
    20,
    150,
    n_samples
)

support_calls = np.random.randint(
    0,
    12,
    n_samples
)

# ----------------------------------------------------------
# Generate churn probability.
#
# Higher charges and more support calls increase churn.
# Longer tenure decreases churn.
# ----------------------------------------------------------

logit = (

    -4.0

    + 0.025 * monthly_charges

    + 0.25 * support_calls

    - 0.025 * tenure

    + 0.01 * (age - 40)

)

churn_probability = (
    1 /
    (
        1 +
        np.exp(-logit)
    )
)

churn = (
    np.random.random(n_samples)
    < churn_probability
).astype(int)

df = pd.DataFrame({

    "Age": age,

    "Income": income,

    "Tenure": tenure,

    "MonthlyCharges": monthly_charges,

    "SupportCalls": support_calls,

    "Churn": churn

})

# ==========================================================
# BASIC INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("DATASET INSPECTION")
print("=" * 80)

print("\nShape:")

print(df.shape)

print("\nFirst Five Rows:")

print(df.head())

print("\nData Types:")

print(df.dtypes)

print("\nMissing Values:")

print(df.isnull().sum())

print("\nClass Distribution:")

print(
    df["Churn"].value_counts()
)

print("\nClass Proportions:")

print(
    df["Churn"]
    .value_counts(
        normalize=True
    )
)

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE / TARGET SEPARATION")
print("=" * 80)

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]

print("\nFeatures:")

print(X.head())

print("\nTarget:")

print(y.head())

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
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_classifier(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train a classifier and calculate
    classification evaluation metrics.
    """

    model.fit(
        X_train,
        y_train
    )

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    # ------------------------------------------------------
    # Probability / decision score
    # ------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        test_scores = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    else:

        test_scores = (
            model.decision_function(
                X_test
            )
        )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        test_scores
    )

    pr_auc = average_precision_score(
        y_test,
        test_scores
    )

    result = {

        "Model":
            model_name,

        "Train_Accuracy":
            train_accuracy,

        "Test_Accuracy":
            test_accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC_AUC":
            roc_auc,

        "PR_AUC":
            pr_auc

    }

    return (
        result,
        test_predictions,
        test_scores
    )


# ==========================================================
# MODEL STORAGE
# ==========================================================

results = []

predictions = {}

scores = {}

models = {}

# ==========================================================
# 1. LOGISTIC REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("1. LOGISTIC REGRESSION")
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
                max_iter=2000
            )
        )

    ]

)

result, prediction, score = (
    evaluate_classifier(

        "Logistic Regression",

        logistic_model,

        X_train,
        X_test,

        y_train,
        y_test

    )
)

results.append(result)

predictions[
    "Logistic Regression"
] = prediction

scores[
    "Logistic Regression"
] = score

models[
    "Logistic Regression"
] = logistic_model

# ==========================================================
# 2. K-NEAREST NEIGHBORS
# ==========================================================

print("\n" + "=" * 80)
print("2. K-NEAREST NEIGHBORS")
print("=" * 80)

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

result, prediction, score = (
    evaluate_classifier(

        "KNN",

        knn_model,

        X_train,
        X_test,

        y_train,
        y_test

    )
)

results.append(result)

predictions[
    "KNN"
] = prediction

scores[
    "KNN"
] = score

models[
    "KNN"
] = knn_model

# ==========================================================
# 3. DECISION TREE
# ==========================================================

print("\n" + "=" * 80)
print("3. DECISION TREE")
print("=" * 80)

tree_model = DecisionTreeClassifier(

    criterion="gini",

    max_depth=5,

    min_samples_leaf=5,

    random_state=42

)

result, prediction, score = (
    evaluate_classifier(

        "Decision Tree",

        tree_model,

        X_train,
        X_test,

        y_train,
        y_test

    )
)

results.append(result)

predictions[
    "Decision Tree"
] = prediction

scores[
    "Decision Tree"
] = score

models[
    "Decision Tree"
] = tree_model

# ==========================================================
# 4. RANDOM FOREST
# ==========================================================

print("\n" + "=" * 80)
print("4. RANDOM FOREST")
print("=" * 80)

forest_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=8,

    min_samples_leaf=3,

    random_state=42,

    n_jobs=-1

)

result, prediction, score = (
    evaluate_classifier(

        "Random Forest",

        forest_model,

        X_train,
        X_test,

        y_train,
        y_test

    )
)

results.append(result)

predictions[
    "Random Forest"
] = prediction

scores[
    "Random Forest"
] = score

models[
    "Random Forest"
] = forest_model

# ==========================================================
# 5. SUPPORT VECTOR MACHINE
# ==========================================================

print("\n" + "=" * 80)
print("5. SUPPORT VECTOR MACHINE")
print("=" * 80)

svm_model = Pipeline(

    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            SVC(

                kernel="rbf",

                C=1.0,

                gamma="scale",

                probability=True

            )
        )

    ]

)

result, prediction, score = (
    evaluate_classifier(

        "SVM",

        svm_model,

        X_train,
        X_test,

        y_train,
        y_test

    )
)

results.append(result)

predictions[
    "SVM"
] = prediction

scores[
    "SVM"
] = score

models[
    "SVM"
] = svm_model

# ==========================================================
# 6. NAIVE BAYES
# ==========================================================

print("\n" + "=" * 80)
print("6. NAIVE BAYES")
print("=" * 80)

naive_bayes_model = GaussianNB()

result, prediction, score = (
    evaluate_classifier(

        "Gaussian Naive Bayes",

        naive_bayes_model,

        X_train,
        X_test,

        y_train,
        y_test

    )
)

results.append(result)

predictions[
    "Gaussian Naive Bayes"
] = prediction

scores[
    "Gaussian Naive Bayes"
] = score

models[
    "Gaussian Naive Bayes"
] = naive_bayes_model

# ==========================================================
# MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "F1",
    ascending=False
)

print(
    results_df.to_string(
        index=False
    )
)

# ==========================================================
# BEST MODEL
# ==========================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_predictions = (
    predictions[
        best_model_name
    ]
)

best_scores = (
    scores[
        best_model_name
    ]
)

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    "Best Model based on F1:",
    best_model_name
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

        best_predictions,

        target_names=[
            "No Churn",
            "Churn"
        ],

        zero_division=0

    )
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,

    display_labels=[
        "No Churn",
        "Churn"
    ]
)

disp.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================================
# THRESHOLD ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("DECISION THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

threshold_results = []

for threshold in thresholds:

    threshold_predictions = (
        best_scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_results.append({

        "Threshold":
            threshold,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1

    })

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False
    )
)

print("""
Lowering the threshold generally makes
the classifier more willing to predict
the positive class.

This can increase recall but may also
increase false positives.

The appropriate threshold depends on
business requirements.
""")

# ==========================================================
# ROC CURVES
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

for model_name, model_score in scores.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        model_score
    )

    auc = roc_auc_score(
        y_test,
        model_score
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "roc_curves.png",
    dpi=300
)

plt.show()

# ==========================================================
# PRECISION-RECALL CURVES
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

for model_name, model_score in scores.items():

    precision_curve, recall_curve, _ = (
        precision_recall_curve(

            y_test,

            model_score

        )
    )

    pr_auc = average_precision_score(
        y_test,
        model_score
    )

    plt.plot(

        recall_curve,

        precision_curve,

        label=f"{model_name} (AP={pr_auc:.3f})"

    )

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curves"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "precision_recall_curves.png",
    dpi=300
)

plt.show()

# ==========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        forest_model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print(
    feature_importance.to_string(
        index=False
    )
)

# ==========================================================
# OVERFITTING CHECK
# ==========================================================

print("\n" + "=" * 80)
print("OVERFITTING CHECK")
print("=" * 80)

for _, row in results_df.iterrows():

    gap = (
        row["Train_Accuracy"]
        - row["Test_Accuracy"]
    )

    print(

        f"{row['Model']:<25}"

        f"Train Accuracy = "
        f"{row['Train_Accuracy']:.4f} | "

        f"Test Accuracy = "
        f"{row['Test_Accuracy']:.4f} | "

        f"Gap = {gap:.4f}"

    )

print("""
A large train-test performance gap may
indicate overfitting.

Always evaluate this together with
cross-validation and other metrics.
""")

# ==========================================================
# CLASS WEIGHT CONCEPT
# ==========================================================

print("\n" + "=" * 80)
print("CLASS WEIGHTS")
print("=" * 80)

balanced_model = LogisticRegression(

    class_weight="balanced",

    max_iter=2000

)

balanced_pipeline = Pipeline(

    steps=[

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            balanced_model
        )

    ]

)

balanced_pipeline.fit(
    X_train,
    y_train
)

balanced_predictions = (
    balanced_pipeline.predict(
        X_test
    )
)

print(
    "\nClassification Report with "
    "class_weight='balanced':"
)

print(
    classification_report(

        y_test,

        balanced_predictions,

        target_names=[
            "No Churn",
            "Churn"
        ],

        zero_division=0

    )
)

# ==========================================================
# MODEL CHARACTERISTICS
# ==========================================================

print("\n" + "=" * 80)
print("ALGORITHM SUMMARY")
print("=" * 80)

print("""
Logistic Regression
-------------------
Linear classification model.
Simple, interpretable and provides
probability estimates.

KNN
---
Distance-based algorithm.
Sensitive to feature scaling.

Decision Tree
-------------
Rule-based nonlinear classifier.
Does not require feature scaling.

Random Forest
-------------
Ensemble of decision trees.
Usually more robust than a single tree.

SVM
---
Maximum-margin classifier.
Scaling is generally important.

Naive Bayes
-----------
Probabilistic classifier based on
Bayes theorem and conditional
independence assumptions.
""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Change KNN k values:

   k = 3
   k = 5
   k = 10
   k = 15

2. Change Decision Tree max_depth.

3. Compare Gini and Entropy.

4. Change Random Forest n_estimators.

5. Try SVM with a linear kernel.

6. Try SVM with different C values.

7. Compare all models using:

   Accuracy
   Precision
   Recall
   F1
   ROC-AUC
   PR-AUC

8. Generate confusion matrices.

9. Experiment with classification thresholds.

10. Compare class_weight="balanced"
    against the normal model.

11. Determine which model is most
    appropriate when recall is the
    primary objective.

12. Determine which model is most
    appropriate when false positives
    are very costly.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Classification
✔ Binary Classification
✔ Logistic Regression
✔ K-Nearest Neighbors
✔ Decision Tree
✔ Random Forest
✔ Support Vector Machine
✔ Naive Bayes
✔ Probability Prediction
✔ Confusion Matrix
✔ Accuracy
✔ Precision
✔ Recall
✔ F1 Score
✔ ROC-AUC
✔ PR-AUC
✔ Decision Threshold
✔ Class Weights
✔ Feature Importance
✔ Overfitting Detection
✔ Model Comparison
""")

print("\nLesson 05 Completed Successfully!")
