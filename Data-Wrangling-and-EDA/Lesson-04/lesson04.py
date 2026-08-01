"""
Lesson 04 : Outlier Detection & Data Transformation

Topics Covered
---------------
1. Z-Score
2. IQR Method
3. Winsorization
4. Isolation Forest
5. Log Transformation
6. Box-Cox Transformation
7. Yeo-Johnson Transformation

Author : Krish Goel
Repository : Summer Training
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.stats.mstats import winsorize

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer

# ==================================================
# LOAD DATASET
# ==================================================

print("=" * 70)
print("OUTLIER DETECTION & DATA TRANSFORMATION")
print("=" * 70)

df = sns.load_dataset("titanic")

# ==================================================
# BOXPLOT
# ==================================================
# Visualize potential outliers.

plt.figure(figsize=(8,4))
plt.boxplot(df["fare"].dropna())
plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.show()

# ==================================================
# Z-SCORE METHOD
# ==================================================
# Detect observations far from the mean.

fare = df["fare"].dropna()

z_scores = np.abs(zscore(fare))

outliers = fare[z_scores > 3]

print("\n========== Z-SCORE ==========")
print("Number of Outliers:", len(outliers))

# ==================================================
# IQR METHOD
# ==================================================
# Detect outliers using quartiles.

Q1 = fare.quantile(0.25)
Q3 = fare.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

iqr_outliers = fare[(fare < lower) | (fare > upper)]

print("\n========== IQR METHOD ==========")
print("Number of Outliers:", len(iqr_outliers))

# ==================================================
# WINSORIZATION
# ==================================================
# Caps extreme values.

winsor_fare = winsorize(fare, limits=[0.05,0.05])

print("\n========== WINSORIZATION ==========")
print(winsor_fare[:10])

# ==================================================
# ISOLATION FOREST
# ==================================================
# Machine Learning based anomaly detection.

iso = IsolationForest(random_state=42)

prediction = iso.fit_predict(fare.to_frame())

print("\n========== ISOLATION FOREST ==========")
print(pd.Series(prediction).value_counts())

# ==================================================
# LOG TRANSFORMATION
# ==================================================
# Reduces positive skew.

log_fare = np.log1p(fare)

print("\n========== LOG TRANSFORMATION ==========")
print(log_fare.head())

# ==================================================
# BOX-COX TRANSFORMATION
# ==================================================
# Works only with positive values.

boxcox = PowerTransformer(method="box-cox")

boxcox_data = boxcox.fit_transform(fare.values.reshape(-1,1))

print("\n========== BOX-COX ==========")
print(boxcox_data[:5])

# ==================================================
# YEO-JOHNSON
# ==================================================
# Handles positive and negative values.

yeo = PowerTransformer(method="yeo-johnson")

yeo_data = yeo.fit_transform(fare.values.reshape(-1,1))

print("\n========== YEO-JOHNSON ==========")
print(yeo_data[:5])

# ==================================================
# MINI PRACTICE
# ==================================================

print("\n========== MINI PRACTICE ==========")

print("Maximum Fare :", fare.max())
print("Minimum Fare :", fare.min())
print("Mean Fare :", fare.mean())
print("Median Fare :", fare.median())

print("\nLesson 04 Completed Successfully!")
