"""
Lesson 02 : Data Preprocessing for Machine Learning

Topics Covered
---------------
1. Feature / Target Separation
2. Train-Test Split
3. Missing Value Handling
4. Numerical Imputation
5. Categorical Imputation
6. One-Hot Encoding
7. Feature Scaling
8. StandardScaler
9. MinMaxScaler
10. RobustScaler
11. ColumnTransformer
12. Pipeline
13. Data Leakage
14. Model Integration

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("DATA PREPROCESSING FOR MACHINE LEARNING")
print("=" * 80)

print("""
Raw Machine Learning data may contain:

- Missing values
- Categorical variables
- Different numerical scales
- Invalid values
- Inconsistent formats

Preprocessing converts the raw data into
a form suitable for Machine Learning models.
""")

# ==========================================================
# CREATE SAMPLE DATASET
# ==========================================================

print("\n" + "=" * 80)
print("CREATING DATASET")
print("=" * 80)

data = {

    "Age": [
        22,
        35,
        np.nan,
        28,
        45,
        31,
        52,
        np.nan,
        40,
        27,
        60,
        33,
        29,
        48,
        25,
        38,
        55,
        42,
        np.nan,
        30
    ],

    "Income": [
        25000,
        50000,
        45000,
        np.nan,
        90000,
        55000,
        120000,
        70000,
        np.nan,
        40000,
        150000,
        60000,
        48000,
        95000,
        30000,
        np.nan,
        110000,
        85000,
        65000,
        42000
    ],

    "SpendingScore": [
        75,
        40,
        60,
        80,
        30,
        np.nan,
        20,
        55,
        70,
        85,
        15,
        65,
        90,
        35,
        78,
        50,
        25,
        45,
        72,
        82
    ],

    "Gender": [
        "Male",
        "Female",
        "Female",
        "Male",
        np.nan,
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female"
    ],

    "City": [
        "Delhi",
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Mumbai",
        "Delhi",
        np.nan,
        "Chennai",
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Chennai",
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Delhi",
        "Chennai",
        "Mumbai",
        "Delhi",
        "Bangalore"
    ],

    "Membership": [
        "Gold",
        "Silver",
        "Gold",
        "Bronze",
        "Gold",
        "Silver",
        "Gold",
        np.nan,
        "Silver",
        "Gold",
        "Gold",
        "Bronze",
        "Silver",
        "Gold",
        "Bronze",
        "Silver",
        "Gold",
        "Gold",
        "Silver",
        "Bronze"
    ],

    "Purchased": [
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        1
    ]
}

df = pd.DataFrame(data)

print("\nDataset:")

print(df)

# ==========================================================
# BASIC INSPECTION
# ==========================================================

print("\n" + "=" * 80)
print("DATASET INSPECTION")
print("=" * 80)

print("\nShape:")

print(df.shape)

print("\nData Types:")

print(df.dtypes)

print("\nMissing Values:")

print(df.isnull().sum())

# ==========================================================
# SEPARATE FEATURES AND TARGET
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE / TARGET SEPARATION")
print("=" * 80)

X = df.drop(
    "Purchased",
    axis=1
)

y = df["Purchased"]

print("\nFeatures:")

print(X.head())

print("\nTarget:")

print(y.head())

# ==========================================================
# IDENTIFY FEATURE TYPES
# ==========================================================

numerical_features = [
    "Age",
    "Income",
    "SpendingScore"
]

categorical_features = [
    "Gender",
    "City",
    "Membership"
]

print("\nNumerical Features:")

print(numerical_features)

print("\nCategorical Features:")

print(categorical_features)

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=42,

    stratify=y

)

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)

print("""
Important:

The split is performed BEFORE fitting
the preprocessing transformations.

This prevents information from the
test set from influencing preprocessing.
""")

# ==========================================================
# NUMERICAL PREPROCESSING
# ==========================================================

print("\n" + "=" * 80)
print("NUMERICAL PREPROCESSING")
print("=" * 80)

numerical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )

    ]

)

print("""
Numerical Pipeline:

Missing Values
      ↓
Median Imputation
      ↓
Standard Scaling
""")

# ==========================================================
# CATEGORICAL PREPROCESSING
# ==========================================================

print("\n" + "=" * 80)
print("CATEGORICAL PREPROCESSING")
print("=" * 80)

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)

print("""
Categorical Pipeline:

Missing Values
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
""")

# ==========================================================
# COLUMN TRANSFORMER
# ==========================================================

print("\n" + "=" * 80)
print("COLUMN TRANSFORMER")
print("=" * 80)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            numerical_pipeline,

            numerical_features
        ),

        (
            "categorical",

            categorical_pipeline,

            categorical_features
        )

    ]

)

print("""
ColumnTransformer applies different
preprocessing pipelines to different
groups of features.
""")

# ==========================================================
# FIT PREPROCESSOR
# ==========================================================

preprocessor.fit(
    X_train
)

print("Preprocessor fitted on training data.")

# ==========================================================
# TRANSFORM DATA
# ==========================================================

X_train_transformed = (
    preprocessor.transform(X_train)
)

X_test_transformed = (
    preprocessor.transform(X_test)
)

print(
    "\nTransformed training shape:",
    X_train_transformed.shape
)

print(
    "Transformed testing shape:",
    X_test_transformed.shape
)

# ==========================================================
# WHY handle_unknown="ignore"?
# ==========================================================

print("\n" + "=" * 80)
print("HANDLE UNKNOWN CATEGORIES")
print("=" * 80)

print("""
Suppose training data contains:

Delhi
Mumbai
Chennai

But new testing/deployment data contains:

Kolkata

handle_unknown="ignore"

prevents the encoder from crashing
when an unseen category appears.
""")

# ==========================================================
# STANDARD SCALER
# ==========================================================

print("\n" + "=" * 80)
print("STANDARD SCALER")
print("=" * 80)

scaler = StandardScaler()

sample_values = np.array(
    [
        [10],
        [20],
        [30],
        [40],
        [50]
    ]
)

standardized = scaler.fit_transform(
    sample_values
)

print("\nOriginal Values:")

print(sample_values.flatten())

print("\nStandardized Values:")

print(
    standardized.flatten()
)

print("""
StandardScaler transforms features so that
they are centered around mean 0 with
standard deviation approximately equal to 1.
""")

# ==========================================================
# MINMAX SCALER
# ==========================================================

print("\n" + "=" * 80)
print("MINMAX SCALER")
print("=" * 80)

minmax = MinMaxScaler()

minmax_values = minmax.fit_transform(
    sample_values
)

print("\nMin-Max Scaled Values:")

print(
    minmax_values.flatten()
)

print("""
MinMaxScaler generally transforms values
into the range [0, 1].
""")

# ==========================================================
# ROBUST SCALER
# ==========================================================

print("\n" + "=" * 80)
print("ROBUST SCALER")
print("=" * 80)

robust = RobustScaler()

robust_values = robust.fit_transform(
    sample_values
)

print("\nRobust Scaled Values:")

print(
    robust_values.flatten()
)

print("""
RobustScaler uses statistics based on the
median and interquartile range.

It can be useful when features contain
strong outliers.
""")

# ==========================================================
# COMPLETE ML PIPELINE
# ==========================================================

print("\n" + "=" * 80)
print("COMPLETE ML PIPELINE")
print("=" * 80)

model_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "model",

            LogisticRegression(
                max_iter=1000
            )
        )

    ]

)

print("""
Complete Pipeline:

Raw Data
   ↓
Preprocessor
   ↓
Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
Logistic Regression
""")

# ==========================================================
# TRAIN PIPELINE
# ==========================================================

model_pipeline.fit(
    X_train,
    y_train
)

print(
    "Complete pipeline trained successfully."
)

# ==========================================================
# PREDICTION
# ==========================================================

predictions = model_pipeline.predict(
    X_test
)

probabilities = (
    model_pipeline.predict_proba(
        X_test
    )[:, 1]
)

# ==========================================================
# EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

print(
    f"Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nPredicted Probabilities:")

print(
    probabilities
)

# ==========================================================
# DEMONSTRATE DATA LEAKAGE
# ==========================================================

print("\n" + "=" * 80)
print("DATA LEAKAGE")
print("=" * 80)

print("""
INCORRECT:

Entire Dataset
      ↓
Fit Imputer / Scaler
      ↓
Train-Test Split

The preprocessing step has already
seen information from the test data.

CORRECT:

Entire Dataset
      ↓
Train-Test Split
      ↓
Fit preprocessing on Training Data
      ↓
Transform Training Data
      ↓
Transform Testing Data

Using Pipeline and ColumnTransformer
helps enforce this workflow.
""")

# ==========================================================
# FIT VS TRANSFORM
# ==========================================================

print("\n" + "=" * 80)
print("FIT vs TRANSFORM")
print("=" * 80)

print("""
fit()
-----
Learns parameters from data.

Example:
StandardScaler learns mean and standard deviation.

transform()
-----------
Uses learned parameters to transform data.

fit_transform()
---------------
Performs both operations.

Recommended:

Training:
fit_transform()

Testing:
transform()
""")

# ==========================================================
# DEPLOYMENT CONCEPT
# ==========================================================

print("\n" + "=" * 80)
print("DEPLOYMENT CONCEPT")
print("=" * 80)

print("""
For deployment, save the COMPLETE pipeline.

New Data
   ↓
Saved Pipeline
   ↓
Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
Model
   ↓
Prediction

This ensures that new data receives
exactly the same transformations used
during training.
""")

# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Replace median imputation with mean
   imputation.

2. Use MinMaxScaler instead of StandardScaler.

3. Use RobustScaler and compare results.

4. Add another categorical feature.

5. Add another numerical feature.

6. Experiment with:
   handle_unknown="ignore"

7. Build the preprocessing workflow
   without Pipeline.

8. Rebuild it using Pipeline.

9. Explain why the Pipeline approach
   is safer.

10. Demonstrate an example of data leakage.
""")

# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Feature / Target Separation
✔ Train-Test Split
✔ Stratified Split
✔ Missing Value Imputation
✔ Numerical Preprocessing
✔ Categorical Preprocessing
✔ One-Hot Encoding
✔ StandardScaler
✔ MinMaxScaler
✔ RobustScaler
✔ ColumnTransformer
✔ Pipeline
✔ fit()
✔ transform()
✔ fit_transform()
✔ Data Leakage Prevention
✔ ML Preprocessing Pipeline
""")

print("\nLesson 02 Completed Successfully!")
