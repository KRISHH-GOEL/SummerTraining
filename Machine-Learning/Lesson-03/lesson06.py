"""
Lesson 03 : Feature Engineering for Machine Learning

Topics Covered
---------------
1. Feature Creation
2. Binning
3. One-Hot Encoding
4. Ordinal Encoding
5. Frequency Encoding
6. Target Encoding Concept
7. Log Transformation
8. Yeo-Johnson Transformation
9. Polynomial Features
10. Interaction Features
11. Correlation-Based Feature Selection
12. SelectKBest
13. Mutual Information
14. RFE
15. Class Imbalance
16. Random Undersampling
17. Random Oversampling
18. SMOTE
19. ADASYN
20. Leakage Prevention

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer
)

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    RFE
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN
)

from imblearn.under_sampling import (
    RandomUnderSampler
)

# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("FEATURE ENGINEERING FOR MACHINE LEARNING")
print("=" * 80)

print("""
Feature Engineering transforms raw variables
into meaningful features that can help a
Machine Learning model learn useful patterns.
""")

# ==========================================================
# CREATE DATASET
# ==========================================================

np.random.seed(42)

n = 300

age = np.random.randint(
    18,
    70,
    n
)

income = np.random.randint(
    20000,
    150000,
    n
)

tenure = np.random.randint(
    1,
    72,
    n
)

monthly_charges = np.random.uniform(
    20,
    150,
    n
)

gender = np.random.choice(
    [
        "Male",
        "Female"
    ],
    n
)

contract = np.random.choice(
    [
        "Month-to-month",
        "One year",
        "Two year"
    ],
    n,
    p=[
        0.55,
        0.25,
        0.20
    ]
)

city = np.random.choice(
    [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Chennai",
        "Pune"
    ],
    n
)

# Create an imbalanced target.

churn = np.random.choice(
    [
        0,
        1
    ],
    n,
    p=[
        0.85,
        0.15
    ]
)

df = pd.DataFrame({

    "Age": age,

    "Income": income,

    "Tenure": tenure,

    "MonthlyCharges": monthly_charges,

    "Gender": gender,

    "Contract": contract,

    "City": city,

    "Churn": churn

})

print("\nDataset Shape:")

print(df.shape)

print("\nFirst Five Rows:")

print(df.head())

# ==========================================================
# BASIC INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("BASIC INSPECTION")
print("=" * 80)

print("\nData Types:")

print(df.dtypes)

print("\nMissing Values:")

print(df.isnull().sum())

print("\nClass Distribution:")

print(
    df["Churn"].value_counts()
)

# ==========================================================
# FEATURE CREATION
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE CREATION")
print("=" * 80)

# Total estimated charges.

df["EstimatedTotalCharges"] = (
    df["Tenure"]
    * df["MonthlyCharges"]
)

# Average annual income.

df["AnnualIncome"] = (
    df["Income"]
)

# Income per month.

df["MonthlyIncome"] = (
    df["Income"]
    / 12
)

# Age-income interaction.

df["AgeIncomeRatio"] = (
    df["Income"]
    / df["Age"]
)

print("\nNew Features:")

print(
    df[
        [
            "EstimatedTotalCharges",
            "AnnualIncome",
            "MonthlyIncome",
            "AgeIncomeRatio"
        ]
    ].head()
)

# ==========================================================
# BINNING
# ==========================================================

print("\n" + "=" * 80)
print("BINNING / DISCRETIZATION")
print("=" * 80)

# Equal-width binning.

df["AgeGroup"] = pd.cut(

    df["Age"],

    bins=[
        0,
        18,
        30,
        45,
        60,
        100
    ],

    labels=[
        "Teen",
        "Young Adult",
        "Adult",
        "Middle Aged",
        "Senior"
    ]

)

print("\nEqual-Width Binning:")

print(
    df[
        [
            "Age",
            "AgeGroup"
        ]
    ].head(10)
)

# Equal-frequency binning.

df["IncomeQuantile"] = pd.qcut(

    df["Income"],

    q=4,

    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]

)

print("\nEqual-Frequency Binning:")

print(
    df[
        [
            "Income",
            "IncomeQuantile"
        ]
    ].head(10)
)

# ==========================================================
# ONE-HOT ENCODING
# ==========================================================

print("\n" + "=" * 80)
print("ONE-HOT ENCODING")
print("=" * 80)

one_hot = pd.get_dummies(

    df[
        [
            "Gender",
            "City"
        ]
    ],

    drop_first=True,

    dtype=int

)

print("\nOne-Hot Encoded Features:")

print(one_hot.head())

# ==========================================================
# ORDINAL ENCODING
# ==========================================================

print("\n" + "=" * 80)
print("ORDINAL ENCODING")
print("=" * 80)

contract_order = [
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
]

ordinal_encoder = OrdinalEncoder(
    categories=contract_order
)

contract_encoded = ordinal_encoder.fit_transform(

    df[
        [
            "Contract"
        ]
    ]

)

print("\nOriginal Contract:")

print(
    df["Contract"].head(10).values
)

print("\nEncoded Contract:")

print(
    contract_encoded[:10].flatten()
)

print("""
The order used is:

Month-to-month → 0
One year       → 1
Two year       → 2
""")

# ==========================================================
# FREQUENCY ENCODING
# ==========================================================

print("\n" + "=" * 80)
print("FREQUENCY ENCODING")
print("=" * 80)

city_frequency = (
    df["City"]
    .value_counts(
        normalize=True
    )
)

df["CityFrequency"] = (
    df["City"]
    .map(city_frequency)
)

print(
    df[
        [
            "City",
            "CityFrequency"
        ]
    ].head(10)
)

# ==========================================================
# TARGET ENCODING CONCEPT
# ==========================================================

print("\n" + "=" * 80)
print("TARGET ENCODING CONCEPT")
print("=" * 80)

target_encoding = (

    df.groupby("City")["Churn"]
    .mean()

)

print("\nTarget Mean by City:")

print(target_encoding)

print("""
Important:

Target encoding uses the target variable.

Therefore, calculating it using the complete
dataset before splitting can cause target leakage.

In production ML workflows, use a
leakage-safe / cross-validated approach.
""")

# ==========================================================
# LOG TRANSFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("LOG TRANSFORMATION")
print("=" * 80)

df["LogIncome"] = np.log1p(
    df["Income"]
)

df["LogTotalCharges"] = np.log1p(
    df["EstimatedTotalCharges"]
)

print(
    df[
        [
            "Income",
            "LogIncome"
        ]
    ].head()
)

print("""
Log transformations can reduce strong
right-skewness in positive numerical features.
""")

# ==========================================================
# YEO-JOHNSON TRANSFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("YEO-JOHNSON TRANSFORMATION")
print("=" * 80)

power_transformer = PowerTransformer(
    method="yeo-johnson"
)

df["Income_YeoJohnson"] = (
    power_transformer.fit_transform(
        df[
            [
                "Income"
            ]
        ]
    )
)

print(
    df[
        [
            "Income",
            "Income_YeoJohnson"
        ]
    ].head()
)

print("""
Yeo-Johnson can handle zero and negative
values unlike Box-Cox.
""")

# ==========================================================
# POLYNOMIAL FEATURES
# ==========================================================

print("\n" + "=" * 80)
print("POLYNOMIAL FEATURES")
print("=" * 80)

small_data = pd.DataFrame({

    "Age": [
        20,
        30,
        40
    ],

    "Income": [
        30000,
        50000,
        70000
    ]

})

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

poly_features = poly.fit_transform(
    small_data
)

poly_names = poly.get_feature_names_out(
    small_data.columns
)

poly_df = pd.DataFrame(
    poly_features,
    columns=poly_names
)

print("\nPolynomial Features:")

print(poly_df)

print("""
Degree 2 creates:

Age
Income
Age²
Age × Income
Income²
""")

# ==========================================================
# TRAIN-TEST SPLIT FOR FEATURE SELECTION
# ==========================================================

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

feature_columns = [

    "Age",
    "Income",
    "Tenure",
    "MonthlyCharges",
    "EstimatedTotalCharges",
    "AnnualIncome",
    "MonthlyIncome",
    "AgeIncomeRatio"

]

X = df[
    feature_columns
]

y = df["Churn"]

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,
        y,

        test_size=0.25,

        random_state=42,

        stratify=y

    )
)

print(
    "Training shape:",
    X_train.shape
)

print(
    "Testing shape:",
    X_test.shape
)

# ==========================================================
# CORRELATION-BASED FEATURE SELECTION
# ==========================================================

print("\n" + "=" * 80)
print("CORRELATION-BASED FEATURE SELECTION")
print("=" * 80)

correlation = (

    X_train
    .corrwith(y_train)
    .abs()
    .sort_values(
        ascending=False
    )

)

print("\nAbsolute Feature-Target Correlations:")

print(correlation)

# ==========================================================
# SELECTKBEST
# ==========================================================

print("\n" + "=" * 80)
print("SELECTKBEST")
print("=" * 80)

k = 5

selector = SelectKBest(

    score_func=mutual_info_classif,

    k=k

)

X_train_selected = selector.fit_transform(
    X_train,
    y_train
)

selected_features = (
    X_train.columns[
        selector.get_support()
    ]
)

print("\nSelected Features:")

print(
    selected_features.tolist()
)

print(
    "\nTransformed Shape:",
    X_train_selected.shape
)

# ==========================================================
# MUTUAL INFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("MUTUAL INFORMATION")
print("=" * 80)

mi_scores = mutual_info_classif(

    X_train,
    y_train,

    random_state=42

)

mi_results = (

    pd.DataFrame({

        "Feature":
            X_train.columns,

        "MutualInformation":
            mi_scores

    })

    .sort_values(
        "MutualInformation",
        ascending=False
    )

)

print(mi_results)

print("""
Mutual Information measures how much
information a feature provides about
the target.

Unlike simple correlation, it can capture
certain nonlinear dependencies.
""")

# ==========================================================
# RFE
# ==========================================================

print("\n" + "=" * 80)
print("RECURSIVE FEATURE ELIMINATION")
print("=" * 80)

rfe_model = LogisticRegression(
    max_iter=2000
)

rfe = RFE(

    estimator=rfe_model,

    n_features_to_select=5

)

rfe.fit(
    X_train,
    y_train
)

rfe_results = pd.DataFrame({

    "Feature":
        X_train.columns,

    "Selected":
        rfe.support_,

    "Ranking":
        rfe.ranking_

})

print(rfe_results)

# ==========================================================
# CLASS IMBALANCE
# ==========================================================

print("\n" + "=" * 80)
print("CLASS IMBALANCE")
print("=" * 80)

print("\nOriginal Class Distribution:")

print(
    y_train.value_counts()
)

print(
    "\nClass Proportions:"
)

print(
    y_train.value_counts(
        normalize=True
    )
)

print("""
If one class significantly dominates the
other, the dataset may be imbalanced.

Accuracy alone may then provide a misleading
picture of model performance.
""")

# ==========================================================
# RANDOM OVERSAMPLING
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM OVERSAMPLING")
print("=" * 80)

oversampler = RandomOverSampler(
    random_state=42
)

X_over, y_over = oversampler.fit_resample(
    X_train,
    y_train
)

print("\nBefore:")

print(
    y_train.value_counts()
)

print("\nAfter Random Oversampling:")

print(
    y_over.value_counts()
)

# ==========================================================
# RANDOM UNDERSAMPLING
# ==========================================================

print("\n" + "=" * 80)
print("RANDOM UNDERSAMPLING")
print("=" * 80)

undersampler = RandomUnderSampler(
    random_state=42
)

X_under, y_under = (
    undersampler.fit_resample(
        X_train,
        y_train
    )
)

print("\nBefore:")

print(
    y_train.value_counts()
)

print("\nAfter Random Undersampling:")

print(
    y_under.value_counts()
)

# ==========================================================
# SMOTE
# ==========================================================

print("\n" + "=" * 80)
print("SMOTE")
print("=" * 80)

smote = SMOTE(
    random_state=42
)

X_smote, y_smote = (
    smote.fit_resample(
        X_train,
        y_train
    )
)

print("\nBefore SMOTE:")

print(
    y_train.value_counts()
)

print("\nAfter SMOTE:")

print(
    y_smote.value_counts()
)

# ==========================================================
# ADASYN
# ==========================================================

print("\n" + "=" * 80)
print("ADASYN")
print("=" * 80)

adasyn = ADASYN(
    random_state=42
)

try:

    X_adasyn, y_adasyn = (
        adasyn.fit_resample(
            X_train,
            y_train
        )
    )

    print("\nBefore ADASYN:")

    print(
        y_train.value_counts()
    )

    print("\nAfter ADASYN:")

    print(
        y_adasyn.value_counts()
    )

except ValueError as error:

    print(
        "\nADASYN could not generate samples:",
        error
    )

# ==========================================================
# SIMPLE MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 80)
print("IMBALANCED DATA MODEL COMPARISON")
print("=" * 80)

baseline_model = LogisticRegression(
    max_iter=2000
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_predictions = (
    baseline_model.predict(
        X_test
    )
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print(
    "\nBaseline Accuracy:",
    round(
        baseline_accuracy,
        4
    )
)

smote_model = LogisticRegression(
    max_iter=2000
)

smote_model.fit(
    X_smote,
    y_smote
)

smote_predictions = (
    smote_model.predict(
        X_test
    )
)

smote_accuracy = accuracy_score(
    y_test,
    smote_predictions
)

print(
    "SMOTE Accuracy:",
    round(
        smote_accuracy,
        4
    )
)

print("\nBaseline Classification Report:")

print(
    classification_report(
        y_test,
        baseline_predictions
    )
)

print("\nSMOTE Classification Report:")

print(
    classification_report(
        y_test,
        smote_predictions
    )
)

# ==========================================================
# LEAKAGE WARNING
# ==========================================================

print("\n" + "=" * 80)
print("LEAKAGE WARNING")
print("=" * 80)

print("""
DO NOT:

SMOTE
 ↓
Train-Test Split

DO:

Train-Test Split
 ↓
SMOTE on Training Data
 ↓
Train Model
 ↓
Evaluate on untouched Test Data

For cross-validation, use an
imblearn Pipeline so SMOTE is applied
inside each training fold.
""")

# ==========================================================
# FEATURE ENGINEERING PRINCIPLES
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING PRINCIPLES")
print("=" * 80)

print("""
1. Create features based on domain knowledge.

2. Do not create features using future information.

3. Avoid target leakage.

4. Fit learned transformations using training data.

5. Do not blindly create hundreds of features.

6. More features can increase noise and overfitting.

7. Feature selection can improve efficiency
   and interpretability.

8. Evaluate whether an engineered feature
   actually improves model performance.

9. Apply resampling only to training data.

10. Use pipelines for reproducible workflows.
""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Create a new feature from Age and Income.

2. Perform equal-width binning on Income.

3. Perform equal-frequency binning on Age.

4. Apply log transformation to Income.

5. Create polynomial features with degree 2.

6. Compare One-Hot and Ordinal Encoding.

7. Perform frequency encoding on City.

8. Calculate target encoding carefully
   using training data only.

9. Use SelectKBest to select features.

10. Apply RFE.

11. Compare Random Oversampling,
    Random Undersampling, SMOTE and ADASYN.

12. Explain why resampling must happen
    after train-test splitting.

13. Identify three possible examples
    of target leakage.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Feature Creation
✔ Binning
✔ Equal-Width Binning
✔ Equal-Frequency Binning
✔ One-Hot Encoding
✔ Ordinal Encoding
✔ Frequency Encoding
✔ Target Encoding Concept
✔ Log Transformation
✔ Yeo-Johnson Transformation
✔ Polynomial Features
✔ Interaction Features
✔ Correlation-Based Selection
✔ SelectKBest
✔ Mutual Information
✔ RFE
✔ Class Imbalance
✔ Random Oversampling
✔ Random Undersampling
✔ SMOTE
✔ ADASYN
✔ Leakage Prevention
""")

print("\nLesson 03 Completed Successfully!")
