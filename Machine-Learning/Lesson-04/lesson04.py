"""
Lesson 04 : Regression Algorithms

Topics Covered
---------------
1. Regression Fundamentals
2. Train-Test Split
3. Linear Regression
4. Multiple Linear Regression
5. Polynomial Regression
6. Ridge Regression
7. Lasso Regression
8. ElasticNet Regression
9. Decision Tree Regression
10. Random Forest Regression
11. MAE
12. MSE
13. RMSE
14. R2 Score
15. Model Comparison
16. Overfitting Detection
17. Actual vs Predicted Visualization

Dataset
-------
Synthetic House Price Dataset

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

from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# CREATE DATASET
# ==========================================================

print("=" * 80)
print("REGRESSION ALGORITHMS")
print("=" * 80)

np.random.seed(42)

n_samples = 300

area = np.random.randint(
    600,
    3500,
    n_samples
)

bedrooms = np.random.randint(
    1,
    6,
    n_samples
)

bathrooms = np.random.randint(
    1,
    5,
    n_samples
)

age = np.random.randint(
    0,
    30,
    n_samples
)

distance = np.random.uniform(
    1,
    30,
    n_samples
)

# Create a realistic synthetic target.

noise = np.random.normal(
    0,
    25000,
    n_samples
)

price = (

    120000

    + area * 180

    + bedrooms * 35000

    + bathrooms * 25000

    - age * 2500

    - distance * 4000

    + noise

)

df = pd.DataFrame({

    "Area": area,

    "Bedrooms": bedrooms,

    "Bathrooms": bathrooms,

    "Age": age,

    "DistanceFromCity": distance,

    "Price": price

})

print("\nDataset Shape:")

print(df.shape)

print("\nFirst Five Rows:")

print(df.head())

# ==========================================================
# DATA INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("DATA INSPECTION")
print("=" * 80)

print("\nData Types:")

print(df.dtypes)

print("\nMissing Values:")

print(df.isnull().sum())

print("\nDescriptive Statistics:")

print(df.describe())

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE / TARGET SEPARATION")
print("=" * 80)

X = df.drop(
    "Price",
    axis=1
)

y = df["Price"]

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

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

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

def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train a regression model and calculate
    standard regression metrics.
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

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    mse = mean_squared_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(mse)

    return {

        "Model": model_name,

        "Train_R2": train_r2,

        "Test_R2": test_r2,

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse

    }, test_predictions


# ==========================================================
# MODEL STORAGE
# ==========================================================

results = []

predictions_dict = {}

# ==========================================================
# 1. LINEAR REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("1. LINEAR REGRESSION")
print("=" * 80)

linear_model = LinearRegression()

linear_result, linear_predictions = evaluate_model(

    "Linear Regression",

    linear_model,

    X_train,

    X_test,

    y_train,

    y_test

)

results.append(
    linear_result
)

predictions_dict[
    "Linear Regression"
] = linear_predictions

print(
    "\nCoefficients:"
)

print(
    linear_model.coef_
)

print(
    "\nIntercept:",
    linear_model.intercept_
)

# ==========================================================
# 2. POLYNOMIAL REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("2. POLYNOMIAL REGRESSION")
print("=" * 80)

polynomial_model = Pipeline(

    steps=[

        (
            "polynomial_features",

            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),

        (
            "model",

            LinearRegression()
        )

    ]

)

polynomial_result, polynomial_predictions = (
    evaluate_model(

        "Polynomial Regression",

        polynomial_model,

        X_train,

        X_test,

        y_train,

        y_test

    )
)

results.append(
    polynomial_result
)

predictions_dict[
    "Polynomial Regression"
] = polynomial_predictions

# ==========================================================
# 3. RIDGE REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("3. RIDGE REGRESSION")
print("=" * 80)

ridge_model = Pipeline(

    steps=[

        (
            "scaler",

            StandardScaler()
        ),

        (
            "model",

            Ridge(
                alpha=10
            )
        )

    ]

)

ridge_result, ridge_predictions = evaluate_model(

    "Ridge Regression",

    ridge_model,

    X_train,

    X_test,

    y_train,

    y_test

)

results.append(
    ridge_result
)

predictions_dict[
    "Ridge Regression"
] = ridge_predictions

# ==========================================================
# 4. LASSO REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("4. LASSO REGRESSION")
print("=" * 80)

lasso_model = Pipeline(

    steps=[

        (
            "scaler",

            StandardScaler()
        ),

        (
            "model",

            Lasso(
                alpha=100
            )
        )

    ]

)

lasso_result, lasso_predictions = evaluate_model(

    "Lasso Regression",

    lasso_model,

    X_train,

    X_test,

    y_train,

    y_test

)

results.append(
    lasso_result
)

predictions_dict[
    "Lasso Regression"
] = lasso_predictions

# ==========================================================
# 5. ELASTIC NET
# ==========================================================

print("\n" + "=" * 80)
print("5. ELASTIC NET")
print("=" * 80)

elastic_model = Pipeline(

    steps=[

        (
            "scaler",

            StandardScaler()
        ),

        (
            "model",

            ElasticNet(

                alpha=0.1,

                l1_ratio=0.5,

                max_iter=10000

            )

        )

    ]

)

elastic_result, elastic_predictions = evaluate_model(

    "ElasticNet",

    elastic_model,

    X_train,

    X_test,

    y_train,

    y_test

)

results.append(
    elastic_result
)

predictions_dict[
    "ElasticNet"
] = elastic_predictions

# ==========================================================
# 6. DECISION TREE REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("6. DECISION TREE REGRESSION")
print("=" * 80)

tree_model = DecisionTreeRegressor(

    max_depth=6,

    min_samples_leaf=4,

    random_state=42

)

tree_result, tree_predictions = evaluate_model(

    "Decision Tree",

    tree_model,

    X_train,

    X_test,

    y_train,

    y_test

)

results.append(
    tree_result
)

predictions_dict[
    "Decision Tree"
] = tree_predictions

# ==========================================================
# 7. RANDOM FOREST REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("7. RANDOM FOREST REGRESSION")
print("=" * 80)

forest_model = RandomForestRegressor(

    n_estimators=200,

    max_depth=10,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)

forest_result, forest_predictions = evaluate_model(

    "Random Forest",

    forest_model,

    X_train,

    X_test,

    y_train,

    y_test

)

results.append(
    forest_result
)

predictions_dict[
    "Random Forest"
] = forest_predictions

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
    "Test_R2",
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
    predictions_dict[
        best_model_name
    ]
)

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    "Best Model:",
    best_model_name
)

# ==========================================================
# BEST MODEL METRICS
# ==========================================================

best_metrics = results_df[
    results_df["Model"]
    == best_model_name
].iloc[0]

print(
    f"\nTest R² : "
    f"{best_metrics['Test_R2']:.4f}"
)

print(
    f"MAE     : "
    f"{best_metrics['MAE']:.2f}"
)

print(
    f"RMSE    : "
    f"{best_metrics['RMSE']:.2f}"
)

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

print("\n" + "=" * 80)
print("ACTUAL VS PREDICTED")
print("=" * 80)

comparison = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": best_predictions

})

print(
    comparison.head(10)
)

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(

    y_test,

    best_predictions,

    alpha=0.7

)

# Perfect prediction line.

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(

    [minimum, maximum],

    [minimum, maximum]

)

plt.xlabel(
    "Actual Price"
)

plt.ylabel(
    "Predicted Price"
)

plt.title(
    f"Actual vs Predicted – {best_model_name}"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ==========================================================
# RESIDUAL ANALYSIS
# ==========================================================

residuals = (
    y_test.values
    - best_predictions
)

plt.figure(
    figsize=(8, 6)
)

plt.scatter(

    best_predictions,

    residuals,

    alpha=0.7

)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Price"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "residual_plot.png",
    dpi=300
)

plt.show()

print("""
Residual:

Actual - Predicted

A residual plot can help identify
systematic patterns in model errors.
""")

# ==========================================================
# OVERFITTING CHECK
# ==========================================================

print("\n" + "=" * 80)
print("OVERFITTING CHECK")
print("=" * 80)

for _, row in results_df.iterrows():

    gap = (
        row["Train_R2"]
        - row["Test_R2"]
    )

    print(
        f"{row['Model']:<25}"
        f"Train R² = {row['Train_R2']:.4f} | "
        f"Test R² = {row['Test_R2']:.4f} | "
        f"Gap = {gap:.4f}"
    )

print("""
A large difference between training and testing
performance can indicate overfitting.

However, the train-test gap should always be
interpreted together with cross-validation and
the nature of the dataset.
""")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({

    "Feature": X.columns,

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
# MODEL SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("ALGORITHM SUMMARY")
print("=" * 80)

print("""
Linear Regression
-----------------
Simple and interpretable.
Works well when relationships are
approximately linear.

Polynomial Regression
---------------------
Extends linear regression to model
nonlinear relationships.

Ridge
-----
L2 regularization.
Useful when coefficients need to
be controlled.

Lasso
-----
L1 regularization.
Can shrink some coefficients to zero.

ElasticNet
----------
Combines L1 and L2 regularization.

Decision Tree
-------------
Captures nonlinear relationships
using decision rules.

Random Forest
-------------
Combines many decision trees to
improve generalization.
""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Train Linear Regression using
   only Area.

2. Train Multiple Linear Regression
   using all features.

3. Try Polynomial Regression with
   degree 2 and degree 3.

4. Change Ridge alpha:
   0.01, 1, 10, 100.

5. Change Lasso alpha and observe
   coefficient changes.

6. Experiment with ElasticNet's
   l1_ratio.

7. Change Decision Tree max_depth.

8. Change Random Forest n_estimators.

9. Compare all models using:
   MAE
   MSE
   RMSE
   R²

10. Identify the best model.

11. Analyze the residual plot.

12. Compare training and testing R².

13. Determine whether any model
    appears to overfit.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Regression
✔ Linear Regression
✔ Multiple Linear Regression
✔ Polynomial Regression
✔ Ridge Regression
✔ Lasso Regression
✔ ElasticNet
✔ Decision Tree Regression
✔ Random Forest Regression
✔ MAE
✔ MSE
✔ RMSE
✔ R²
✔ Model Comparison
✔ Residual Analysis
✔ Overfitting Detection
✔ Feature Importance
✔ Actual vs Predicted Visualization
""")

print("\nLesson 04 Completed Successfully!")
