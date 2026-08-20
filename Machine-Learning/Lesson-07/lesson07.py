"""
Lesson 07 : Model Evaluation, Validation & Hyperparameter Tuning

Topics Covered
---------------
1. Train / Test Evaluation
2. Overfitting
3. Underfitting
4. K-Fold Cross-Validation
5. Stratified K-Fold
6. Cross-Validation Scores
7. Model Comparison
8. GridSearchCV
9. RandomizedSearchCV
10. Hyperparameter Tuning
11. Classification Metrics
12. Learning Curves
13. Pipelines
14. Data Leakage Prevention

Author : Krish Goel
Repository : Summer Training
"""


# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

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
print("MODEL EVALUATION, VALIDATION & HYPERPARAMETER TUNING")
print("=" * 80)

print("""
The objective of this lesson is to evaluate Machine Learning
models properly, compare their generalization performance,
and improve them using hyperparameter tuning.
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

print("\nFeature Shape:")

print(X.shape)

print("\nTarget Distribution:")

print(
    y.value_counts()
    .sort_index()
)

print("\nFirst Five Rows:")

print(
    X.head()
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
# PIPELINE 1 : LOGISTIC REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION PIPELINE")
print("=" * 80)

logistic_pipeline = Pipeline(

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


# ==========================================================
# PIPELINE 2 : DECISION TREE
# ==========================================================

tree_pipeline = Pipeline(

    steps=[

        (
            "model",
            DecisionTreeClassifier(
                random_state=42
            )
        )

    ]

)


# ==========================================================
# PIPELINE 3 : RANDOM FOREST
# ==========================================================

forest_pipeline = Pipeline(

    steps=[

        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )

    ]

)


# ==========================================================
# SIMPLE TRAIN-TEST EVALUATION
# ==========================================================

print("\n" + "=" * 80)
print("TRAIN-TEST EVALUATION")
print("=" * 80)


models = {

    "Logistic Regression":
        logistic_pipeline,

    "Decision Tree":
        tree_pipeline,

    "Random Forest":
        forest_pipeline

}


simple_results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    train_predictions = (
        model.predict(
            X_train
        )
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )

    train_accuracy = (
        accuracy_score(
            y_train,
            train_predictions
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_predictions
        )
    )

    simple_results.append({

        "Model":
            name,

        "Train Accuracy":
            train_accuracy,

        "Test Accuracy":
            test_accuracy,

        "Gap":
            train_accuracy - test_accuracy

    })


simple_results_df = pd.DataFrame(
    simple_results
)

print(
    simple_results_df.to_string(
        index=False
    )
)


# ==========================================================
# OVERFITTING / UNDERFITTING
# ==========================================================

print("\n" + "=" * 80)
print("OVERFITTING / UNDERFITTING ANALYSIS")
print("=" * 80)

for _, row in simple_results_df.iterrows():

    gap = row["Gap"]

    print(
        f"\n{row['Model']}"
    )

    print(
        f"Training Accuracy : "
        f"{row['Train Accuracy']:.4f}"
    )

    print(
        f"Testing Accuracy  : "
        f"{row['Test Accuracy']:.4f}"
    )

    print(
        f"Train-Test Gap    : "
        f"{gap:.4f}"
    )

    if (
        row["Train Accuracy"] < 0.75
        and row["Test Accuracy"] < 0.75
    ):

        print(
            "Possible indication of underfitting."
        )

    elif gap > 0.10:

        print(
            "Possible indication of overfitting."
        )

    else:

        print(
            "No strong indication from this simple gap check."
        )


# ==========================================================
# K-FOLD CROSS-VALIDATION
# ==========================================================

print("\n" + "=" * 80)
print("K-FOLD CROSS-VALIDATION")
print("=" * 80)

kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


for name, model in models.items():

    cv_scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=kf,

        scoring="accuracy"

    )

    print(
        f"\n{name}"
    )

    print(
        "Fold Scores:",
        np.round(
            cv_scores,
            4
        )
    )

    print(
        "Mean:",
        round(
            cv_scores.mean(),
            4
        )
    )

    print(
        "Standard Deviation:",
        round(
            cv_scores.std(),
            4
        )
    )


# ==========================================================
# STRATIFIED K-FOLD
# ==========================================================

print("\n" + "=" * 80)
print("STRATIFIED K-FOLD CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


for name, model in models.items():

    scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=skf,

        scoring="accuracy"

    )

    print(
        f"\n{name}"
    )

    print(
        "Scores:",
        np.round(
            scores,
            4
        )
    )

    print(
        "Mean:",
        round(
            scores.mean(),
            4
        )
    )

    print(
        "Std:",
        round(
            scores.std(),
            4
        )
    )


# ==========================================================
# MULTIPLE CROSS-VALIDATION METRICS
# ==========================================================

print("\n" + "=" * 80)
print("MULTIPLE CROSS-VALIDATION METRICS")
print("=" * 80)

scoring_metrics = {

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


cv_results = {}


for name, model in models.items():

    result = cross_validate(

        model,

        X_train,

        y_train,

        cv=skf,

        scoring=scoring_metrics,

        return_train_score=True

    )

    cv_results[name] = result

    print(
        f"\n{name}"
    )

    for metric in scoring_metrics:

        mean_score = (
            result[
                f"test_{metric}"
            ].mean()
        )

        std_score = (
            result[
                f"test_{metric}"
            ].std()
        )

        print(

            f"{metric:<10} "

            f"Mean = {mean_score:.4f} | "

            f"Std = {std_score:.4f}"

        )


# ==========================================================
# MODEL COMPARISON TABLE
# ==========================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION MODEL COMPARISON")
print("=" * 80)

comparison_rows = []


for name, result in cv_results.items():

    comparison_rows.append({

        "Model":
            name,

        "Accuracy":
            result[
                "test_accuracy"
            ].mean(),

        "Precision":
            result[
                "test_precision"
            ].mean(),

        "Recall":
            result[
                "test_recall"
            ].mean(),

        "F1":
            result[
                "test_f1"
            ].mean(),

        "ROC-AUC":
            result[
                "test_roc_auc"
            ].mean()

    })


comparison_df = pd.DataFrame(
    comparison_rows
)

print(
    comparison_df
    .sort_values(
        "F1",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# ==========================================================
# GRID SEARCH
# ==========================================================

print("\n" + "=" * 80)
print("GRID SEARCH - RANDOM FOREST")
print("=" * 80)

forest_param_grid = {

    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]

}


grid_search = GridSearchCV(

    estimator=forest_pipeline,

    param_grid=forest_param_grid,

    cv=skf,

    scoring="f1",

    n_jobs=-1,

    refit=True

)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters:")

print(
    grid_search.best_params_
)

print("\nBest CV F1 Score:")

print(
    grid_search.best_score_
)

print("\nBest Estimator:")

print(
    grid_search.best_estimator_
)


# ==========================================================
# GRID SEARCH RESULTS
# ==========================================================

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results = (
    grid_results
    .sort_values(
        "rank_test_score"
    )
)

print("\nTop Grid Search Results:")

print(
    grid_results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ==========================================================
# RANDOMIZED SEARCH
# ==========================================================

print("\n" + "=" * 80)
print("RANDOMIZED SEARCH - RANDOM FOREST")
print("=" * 80)

random_param_distributions = {

    "model__n_estimators":
        randint(
            100,
            500
        ),

    "model__max_depth": [

        None,
        5,
        10,
        15,
        20

    ],

    "model__min_samples_split":
        randint(
            2,
            15
        ),

    "model__min_samples_leaf":
        randint(
            1,
            8
        )

}


random_search = RandomizedSearchCV(

    estimator=forest_pipeline,

    param_distributions=
        random_param_distributions,

    n_iter=20,

    cv=skf,

    scoring="f1",

    random_state=42,

    n_jobs=-1,

    refit=True

)


random_search.fit(
    X_train,
    y_train
)


print("\nBest Randomized Search Parameters:")

print(
    random_search.best_params_
)

print("\nBest Randomized Search CV F1:")

print(
    random_search.best_score_
)


# ==========================================================
# FINAL MODEL EVALUATION
# ==========================================================

print("\n" + "=" * 80)
print("FINAL MODEL EVALUATION")
print("=" * 80)

final_model = grid_search.best_estimator_

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
print("FINAL CLASSIFICATION REPORT")
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

print(
    cm
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Malignant",
        "Benign"
    ]

)

display.plot()

plt.title(
    "Final Model Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "final_confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================================
# LEARNING CURVE
# ==========================================================

print("\n" + "=" * 80)
print("LEARNING CURVE")
print("=" * 80)

train_sizes, train_scores, validation_scores = (
    learning_curve(

        final_model,

        X_train,

        y_train,

        cv=skf,

        scoring="f1",

        train_sizes=np.linspace(
            0.1,
            1.0,
            10
        ),

        n_jobs=-1

    )
)


train_mean = (
    train_scores.mean(
        axis=1
    )
)

train_std = (
    train_scores.std(
        axis=1
    )
)

validation_mean = (
    validation_scores.mean(
        axis=1
    )
)

validation_std = (
    validation_scores.std(
        axis=1
    )
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(

    train_sizes,

    train_mean,

    marker="o",

    label="Training F1"

)

plt.plot(

    train_sizes,

    validation_mean,

    marker="o",

    label="Validation F1"

)

plt.fill_between(

    train_sizes,

    train_mean - train_std,

    train_mean + train_std,

    alpha=0.15

)

plt.fill_between(

    train_sizes,

    validation_mean - validation_std,

    validation_mean + validation_std,

    alpha=0.15

)

plt.xlabel(
    "Training Set Size"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "Learning Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "learning_curve.png",
    dpi=300
)

plt.show()


# ==========================================================
# MODEL SELECTION SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("MODEL SELECTION SUMMARY")
print("=" * 80)

print("""
Model Development Process

1. Split the dataset into training and test sets.

2. Keep the test set untouched during model development.

3. Build preprocessing and model pipelines.

4. Use cross-validation on the training set.

5. Compare multiple models.

6. Select an appropriate evaluation metric.

7. Tune hyperparameters using GridSearchCV
   or RandomizedSearchCV.

8. Select the best estimator based on
   cross-validation performance.

9. Evaluate the selected model once
   on the unseen test set.

10. Inspect the final metrics and
    learning curve.
""")


# ==========================================================
# DATA LEAKAGE EXPLANATION
# ==========================================================

print("\n" + "=" * 80)
print("DATA LEAKAGE")
print("=" * 80)

print("""
Data leakage occurs when information that should not
be available to the model during training influences
the training process.

Examples:

- Scaling the entire dataset before splitting.
- Calculating imputation values using the test set.
- Selecting features using the entire dataset.
- Using the test set repeatedly during model tuning.

Using Pipeline and cross-validation helps ensure that
preprocessing operations are learned only from the
appropriate training folds.
""")


# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Change the number of CV folds.

2. Compare KFold and StratifiedKFold.

3. Compare models using:

   Accuracy
   Precision
   Recall
   F1
   ROC-AUC

4. Tune Logistic Regression's C.

5. Tune Decision Tree:

   max_depth
   min_samples_split
   min_samples_leaf

6. Expand the Random Forest parameter grid.

7. Compare GridSearchCV and RandomizedSearchCV.

8. Try different scoring metrics.

9. Plot learning curves for multiple models.

10. Identify whether a model is underfitting
    or overfitting.

11. Evaluate the final model only once on
    the untouched test set.

12. Explain why repeatedly checking the test
    set during tuning can introduce bias.
""")


# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Train-Test Evaluation
✔ Generalization
✔ Underfitting
✔ Overfitting
✔ K-Fold Cross-Validation
✔ Stratified K-Fold
✔ Cross-Validation Metrics
✔ Model Comparison
✔ Hyperparameters
✔ GridSearchCV
✔ RandomizedSearchCV
✔ Hyperparameter Tuning
✔ Pipelines
✔ Data Leakage Prevention
✔ Learning Curves
✔ Classification Metrics
✔ Final Model Evaluation
""")

print("\nLesson 07 Completed Successfully!")
