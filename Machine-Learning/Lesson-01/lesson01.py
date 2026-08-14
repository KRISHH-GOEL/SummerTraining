"""
Lesson 01 : Machine Learning Fundamentals

Topics Covered
---------------
1. Machine Learning Introduction
2. Features and Target
3. Supervised Learning
4. Regression
5. Classification
6. Train-Test Split
7. Model Training
8. Prediction
9. Underfitting
10. Overfitting
11. Bias-Variance Concept
12. Basic ML Workflow

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("MACHINE LEARNING FUNDAMENTALS")
print("=" * 80)

print("""
Machine Learning allows computers to learn patterns
from data and use those patterns to make predictions
or decisions.

Basic workflow:

Data
 ↓
Learning Algorithm
 ↓
Model
 ↓
Prediction
""")

# ==========================================================
# AI vs ML vs DEEP LEARNING
# ==========================================================

print("\n" + "=" * 80)
print("AI vs ML vs DEEP LEARNING")
print("=" * 80)

print("""
Artificial Intelligence
-----------------------
Broad field of building systems capable of
performing tasks that normally require intelligence.

Machine Learning
----------------
A subset of AI where systems learn patterns
from data.

Deep Learning
-------------
A subset of Machine Learning based primarily
on multi-layer neural networks.

Relationship:

Artificial Intelligence
        ↓
Machine Learning
        ↓
Deep Learning
""")

# ==========================================================
# TYPES OF MACHINE LEARNING
# ==========================================================

print("\n" + "=" * 80)
print("TYPES OF MACHINE LEARNING")
print("=" * 80)

print("""
1. Supervised Learning
   → Learns from labeled data.

2. Unsupervised Learning
   → Finds patterns in unlabeled data.

3. Semi-Supervised Learning
   → Uses labeled + unlabeled data.

4. Reinforcement Learning
   → Learns through rewards and penalties.
""")

# ==========================================================
# SUPERVISED LEARNING
# ==========================================================

print("\n" + "=" * 80)
print("SUPERVISED LEARNING")
print("=" * 80)

print("""
Supervised Learning:

Features + Known Target
          ↓
       Algorithm
          ↓
        Model
          ↓
     Prediction
""")

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 80)
print("FEATURES AND TARGET")
print("=" * 80)

data = pd.DataFrame({

    "Area": [
        1000,
        1200,
        1500,
        1800,
        2000,
        2500
    ],

    "Bedrooms": [
        2,
        2,
        3,
        3,
        4,
        4
    ],

    "Price": [
        200000,
        240000,
        300000,
        360000,
        410000,
        500000
    ]

})

print(data)

# Features

X = data[
    [
        "Area",
        "Bedrooms"
    ]
]

# Target

y = data["Price"]

print("\nFeatures (X):")

print(X)

print("\nTarget (y):")

print(y)

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.33,

    random_state=42

)

print("Training Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

print("""
Why split the dataset?

The model should be evaluated on data
that it has not seen during training.

This gives a better estimate of
generalization performance.
""")

# ==========================================================
# LINEAR REGRESSION
# ==========================================================

print("\n" + "=" * 80)
print("BASIC REGRESSION MODEL")
print("=" * 80)

model = LinearRegression()

# Train

model.fit(
    X_train,
    y_train
)

print("Model trained successfully.")

# ==========================================================
# PREDICTION
# ==========================================================

predictions = model.predict(
    X_test
)

print("\nActual Values:")

print(y_test.values)

print("\nPredicted Values:")

print(predictions)

# ==========================================================
# MODEL PARAMETERS
# ==========================================================

print("\nModel Coefficients:")

print(model.coef_)

print("\nModel Intercept:")

print(model.intercept_)

print("""
The coefficients represent the estimated
relationship between the input features
and the target variable.
""")

# ==========================================================
# REGRESSION VS CLASSIFICATION
# ==========================================================

print("\n" + "=" * 80)
print("REGRESSION VS CLASSIFICATION")
print("=" * 80)

print("""
REGRESSION
----------

Output:
Continuous numerical value.

Examples:
• House Price
• Salary
• Temperature
• Sales


CLASSIFICATION
--------------

Output:
Discrete class/category.

Examples:
• Spam / Not Spam
• Disease / No Disease
• Churn / No Churn
""")

# ==========================================================
# CLASSIFICATION DATA
# ==========================================================

print("\n" + "=" * 80)
print("CLASSIFICATION EXAMPLE")
print("=" * 80)

classification_data = pd.DataFrame({

    "Age": [
        20,
        22,
        25,
        30,
        35,
        40,
        45,
        50
    ],

    "Income": [
        20000,
        25000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000
    ],

    "Purchased": [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ]

})

print(classification_data)

X_class = classification_data[
    [
        "Age",
        "Income"
    ]
]

y_class = classification_data[
    "Purchased"
]

X_train_class, X_test_class, y_train_class, y_test_class = (
    train_test_split(
        X_class,
        y_class,
        test_size=0.25,
        random_state=42
    )
)

classifier = LogisticRegression()

classifier.fit(
    X_train_class,
    y_train_class
)

class_predictions = classifier.predict(
    X_test_class
)

print("\nActual Classes:")

print(y_test_class.values)

print("\nPredicted Classes:")

print(class_predictions)

# ==========================================================
# PREDICTING PROBABILITY
# ==========================================================

probabilities = classifier.predict_proba(
    X_test_class
)

print("\nPredicted Probabilities:")

print(probabilities)

print("""
Classification models can often provide
probabilities in addition to predicted classes.

For binary classification:

Column 1 → Probability of Class 0
Column 2 → Probability of Class 1
""")

# ==========================================================
# UNDERFITTING AND OVERFITTING
# ==========================================================

print("\n" + "=" * 80)
print("UNDERFITTING AND OVERFITTING")
print("=" * 80)

# Generate synthetic data

np.random.seed(42)

X_curve = np.linspace(
    0,
    10,
    30
)

y_curve = (
    3 * X_curve
    + 5
    + np.random.normal(
        0,
        5,
        size=30
    )
)

X_curve = X_curve.reshape(-1, 1)

X_train_curve, X_test_curve, y_train_curve, y_test_curve = (
    train_test_split(
        X_curve,
        y_curve,
        test_size=0.25,
        random_state=42
    )
)

# ==========================================================
# SIMPLE MODEL
# ==========================================================

simple_model = LinearRegression()

simple_model.fit(
    X_train_curve,
    y_train_curve
)

simple_train_score = simple_model.score(
    X_train_curve,
    y_train_curve
)

simple_test_score = simple_model.score(
    X_test_curve,
    y_test_curve
)

print("\nSimple Linear Model")

print(
    "Training R² :",
    round(simple_train_score, 4)
)

print(
    "Testing R²  :",
    round(simple_test_score, 4)
)

# ==========================================================
# HIGH-COMPLEXITY MODEL
# ==========================================================

complex_model = make_pipeline(

    PolynomialFeatures(
        degree=15
    ),

    LinearRegression()

)

complex_model.fit(
    X_train_curve,
    y_train_curve
)

complex_train_score = complex_model.score(
    X_train_curve,
    y_train_curve
)

complex_test_score = complex_model.score(
    X_test_curve,
    y_test_curve
)

print("\nHigh-Complexity Polynomial Model")

print(
    "Training R² :",
    round(complex_train_score, 4)
)

print(
    "Testing R²  :",
    round(complex_test_score, 4)
)

print("""
Interpretation
--------------

If:

Training performance is very high
BUT
Testing performance is significantly lower

→ The model may be overfitting.

If:

Training performance is poor
AND
Testing performance is also poor

→ The model may be underfitting.

The objective is not to maximize
training performance alone.

The objective is to generalize well
to unseen data.
""")

# ==========================================================
# BIAS-VARIANCE TRADEOFF
# ==========================================================

print("\n" + "=" * 80)
print("BIAS-VARIANCE TRADEOFF")
print("=" * 80)

print("""
High Bias
---------
Model is too simple.

Typical result:
Underfitting.


High Variance
-------------
Model is too sensitive to
training data.

Typical result:
Overfitting.


Goal
----
Find a model complexity that
generalizes well to unseen data.
""")

# ==========================================================
# GENERALIZATION
# ==========================================================

print("\n" + "=" * 80)
print("GENERALIZATION")
print("=" * 80)

print("""
Generalization means that a model
performs well not only on training
data but also on previously unseen data.

This is one of the central goals
of Machine Learning.

A model that memorizes training data
without learning general patterns
will perform poorly on new observations.
""")

# ==========================================================
# COMPLETE MACHINE LEARNING WORKFLOW
# ==========================================================

print("\n" + "=" * 80)
print("COMPLETE MACHINE LEARNING WORKFLOW")
print("=" * 80)

workflow = [

    "1. Define the Problem",

    "2. Collect Data",

    "3. Understand the Data",

    "4. Clean the Data",

    "5. Perform EDA",

    "6. Engineer Features",

    "7. Separate Features and Target",

    "8. Split Training and Testing Data",

    "9. Preprocess Features",

    "10. Select Model",

    "11. Train Model",

    "12. Make Predictions",

    "13. Evaluate Model",

    "14. Tune Hyperparameters",

    "15. Compare Models",

    "16. Save Final Model",

    "17. Deploy Model",

    "18. Monitor Model"

]

for step in workflow:

    print(step)

# ==========================================================
# IMPORTANT PRINCIPLES
# ==========================================================

print("\n" + "=" * 80)
print("IMPORTANT ML PRINCIPLES")
print("=" * 80)

print("""
1. Never evaluate a model only on training data.

2. Keep test data separate until final evaluation.

3. Avoid data leakage.

4. Feature quality matters.

5. Model complexity should be controlled.

6. Choose evaluation metrics based on the problem.

7. Cross-validation gives a more reliable estimate
   of model performance.

8. A more complex model is not automatically better.

9. Good generalization is more important than
   memorizing the training dataset.
""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Identify X and y from a house-price dataset.

2. Identify whether a problem is regression
   or classification.

3. Perform a train-test split.

4. Train a Linear Regression model.

5. Train a Logistic Regression model.

6. Compare training and testing performance.

7. Explain an example of overfitting.

8. Explain an example of underfitting.

9. Explain the difference between:
   AI, ML and Deep Learning.

10. Write the complete ML workflow
    from memory.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Machine Learning Fundamentals
✔ AI vs ML vs Deep Learning
✔ Types of Machine Learning
✔ Supervised Learning
✔ Unsupervised Learning
✔ Features and Target
✔ Regression
✔ Classification
✔ Train-Test Split
✔ Model Training
✔ Prediction
✔ Probability Prediction
✔ Generalization
✔ Underfitting
✔ Overfitting
✔ Bias-Variance Tradeoff
✔ Complete ML Workflow
""")

print("\nLesson 01 Completed Successfully!")
