"""
Lesson 06 : Unsupervised Learning & Clustering

Topics Covered
---------------
1. Unsupervised Learning
2. Clustering
3. Feature Scaling
4. K-Means Clustering
5. Inertia
6. Elbow Method
7. Silhouette Score
8. Agglomerative Clustering
9. Hierarchical Clustering
10. DBSCAN
11. Core / Border / Noise Points
12. Cluster Evaluation
13. Cluster Visualization
14. Customer Segmentation
15. Business Interpretation

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)


# ==========================================================
# INTRODUCTION
# ==========================================================

print("=" * 80)
print("UNSUPERVISED LEARNING & CLUSTERING")
print("=" * 80)

print("""
Unlike supervised learning, clustering does not use
a predefined target variable.

The objective is to discover naturally occurring
groups within the dataset.
""")


# ==========================================================
# CREATE CUSTOMER DATASET
# ==========================================================

np.random.seed(42)

n_samples = 300

# ----------------------------------------------------------
# Generate three broad customer groups.
#
# Group 1 → Lower income / lower spending
# Group 2 → Higher income / higher spending
# Group 3 → Moderate income / moderate spending
# ----------------------------------------------------------

group_1_size = 100
group_2_size = 100
group_3_size = 100


# Group 1

age_1 = np.random.normal(
    30,
    5,
    group_1_size
)

income_1 = np.random.normal(
    35000,
    7000,
    group_1_size
)

spending_1 = np.random.normal(
    30,
    8,
    group_1_size
)

frequency_1 = np.random.normal(
    5,
    2,
    group_1_size
)


# Group 2

age_2 = np.random.normal(
    40,
    6,
    group_2_size
)

income_2 = np.random.normal(
    100000,
    15000,
    group_2_size
)

spending_2 = np.random.normal(
    80,
    8,
    group_2_size
)

frequency_2 = np.random.normal(
    15,
    3,
    group_2_size
)


# Group 3

age_3 = np.random.normal(
    50,
    7,
    group_3_size
)

income_3 = np.random.normal(
    65000,
    10000,
    group_3_size
)

spending_3 = np.random.normal(
    50,
    8,
    group_3_size
)

frequency_3 = np.random.normal(
    9,
    2,
    group_3_size
)


# ==========================================================
# COMBINE DATA
# ==========================================================

age = np.concatenate([
    age_1,
    age_2,
    age_3
])

income = np.concatenate([
    income_1,
    income_2,
    income_3
])

spending = np.concatenate([
    spending_1,
    spending_2,
    spending_3
])

frequency = np.concatenate([
    frequency_1,
    frequency_2,
    frequency_3
])


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame({

    "Age": age,

    "AnnualIncome": income,

    "SpendingScore": spending,

    "PurchaseFrequency": frequency

})


# ==========================================================
# BASIC CLEANUP
# ==========================================================

df["Age"] = df["Age"].round().clip(
    18,
    80
)

df["AnnualIncome"] = (
    df["AnnualIncome"]
    .clip(
        10000,
        200000
    )
)

df["SpendingScore"] = (
    df["SpendingScore"]
    .clip(
        0,
        100
    )
)

df["PurchaseFrequency"] = (
    df["PurchaseFrequency"]
    .clip(
        0,
        30
    )
)


# ==========================================================
# DATA INSPECTION
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

print("\nDescriptive Statistics:")

print(df.describe())


# ==========================================================
# FEATURE SELECTION
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

X = df[
    [
        "Age",
        "AnnualIncome",
        "SpendingScore",
        "PurchaseFrequency"
    ]
]

print("\nFeatures used for clustering:")

print(X.head())


# ==========================================================
# FEATURE SCALING
# ==========================================================

print("\n" + "=" * 80)
print("FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled Features:")

print(
    X_scaled_df.head()
)

print("""
Scaling is important because clustering algorithms
such as K-Means and DBSCAN rely heavily on distances.
""")


# ==========================================================
# ELBOW METHOD
# ==========================================================

print("\n" + "=" * 80)
print("ELBOW METHOD")
print("=" * 80)

inertia_values = []

k_values = range(
    2,
    11
)

for k in k_values:

    model = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    model.fit(
        X_scaled
    )

    inertia_values.append(
        model.inertia_
    )

    print(
        f"K = {k:<3} "
        f"Inertia = {model.inertia_:.2f}"
    )


# ==========================================================
# ELBOW PLOT
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(

    list(k_values),

    inertia_values,

    marker="o"

)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "elbow_method.png",
    dpi=300
)

plt.show()


# ==========================================================
# SILHOUETTE SCORE
# ==========================================================

print("\n" + "=" * 80)
print("SILHOUETTE SCORE")
print("=" * 80)

silhouette_values = []

for k in k_values:

    model = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(

        X_scaled,

        labels

    )

    silhouette_values.append(
        score
    )

    print(
        f"K = {k:<3} "
        f"Silhouette Score = {score:.4f}"
    )


# ==========================================================
# SILHOUETTE PLOT
# ==========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(

    list(k_values),

    silhouette_values,

    marker="o"

)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score vs Number of Clusters"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "silhouette_scores.png",
    dpi=300
)

plt.show()


# ==========================================================
# SELECT K
# ==========================================================

best_k_by_silhouette = (
    list(k_values)[
        np.argmax(
            silhouette_values
        )
    ]
)

print("\n" + "=" * 80)
print("K SELECTION")
print("=" * 80)

print(
    "Best K according to Silhouette Score:",
    best_k_by_silhouette
)

print("""
The silhouette score provides a quantitative
suggestion for K.

The final K should also be evaluated using:

- Elbow curve
- Cluster sizes
- Cluster characteristics
- Visualization
- Domain knowledge
""")


# ==========================================================
# K-MEANS WITH SELECTED K
# ==========================================================

# For this synthetic dataset, K=3 represents
# the intended broad customer groups.

kmeans = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)

kmeans_labels = kmeans.fit_predict(
    X_scaled
)

df["KMeansCluster"] = (
    kmeans_labels
)


# ==========================================================
# K-MEANS RESULTS
# ==========================================================

print("\n" + "=" * 80)
print("K-MEANS RESULTS")
print("=" * 80)

print("\nCluster Counts:")

print(
    df["KMeansCluster"]
    .value_counts()
    .sort_index()
)

print("\nFinal Inertia:")

print(
    kmeans.inertia_
)

print("\nSilhouette Score:")

print(
    silhouette_score(
        X_scaled,
        kmeans_labels
    )
)


# ==========================================================
# CLUSTER PROFILE
# ==========================================================

print("\n" + "=" * 80)
print("K-MEANS CLUSTER PROFILE")
print("=" * 80)

cluster_profile = (
    df
    .groupby(
        "KMeansCluster"
    )
    [
        [
            "Age",
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency"
        ]
    ]
    .mean()
)

print(
    cluster_profile
)


# ==========================================================
# 2D CLUSTER VISUALIZATION
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(

    df["AnnualIncome"],

    df["SpendingScore"],

    c=df["KMeansCluster"],

    alpha=0.7

)

centroids_original_scale = (
    scaler.inverse_transform(
        kmeans.cluster_centers_
    )
)

plt.scatter(

    centroids_original_scale[:, 1],

    centroids_original_scale[:, 2],

    marker="X",

    s=200,

    label="Centroids"

)

plt.xlabel(
    "Annual Income"
)

plt.ylabel(
    "Spending Score"
)

plt.title(
    "K-Means Customer Segmentation"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "kmeans_clusters.png",
    dpi=300
)

plt.show()


# ==========================================================
# AGGLOMERATIVE CLUSTERING
# ==========================================================

print("\n" + "=" * 80)
print("AGGLOMERATIVE CLUSTERING")
print("=" * 80)

hierarchical_model = (
    AgglomerativeClustering(

        n_clusters=3,

        linkage="ward"

    )
)

hierarchical_labels = (
    hierarchical_model
    .fit_predict(
        X_scaled
    )
)

df["HierarchicalCluster"] = (
    hierarchical_labels
)

hierarchical_score = (
    silhouette_score(

        X_scaled,

        hierarchical_labels

    )
)

print(
    "Silhouette Score:",
    hierarchical_score
)

print("\nCluster Counts:")

print(
    df["HierarchicalCluster"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# HIERARCHICAL CLUSTER PROFILE
# ==========================================================

print("\nCluster Profile:")

hierarchical_profile = (
    df
    .groupby(
        "HierarchicalCluster"
    )
    [
        [
            "Age",
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency"
        ]
    ]
    .mean()
)

print(
    hierarchical_profile
)


# ==========================================================
# DENDROGRAM
# ==========================================================

print("\n" + "=" * 80)
print("DENDROGRAM")
print("=" * 80)

# Use a sample for a readable dendrogram.

sample_size = min(
    80,
    len(X_scaled)
)

sample_indices = np.random.choice(

    len(X_scaled),

    sample_size,

    replace=False

)

linkage_matrix = linkage(

    X_scaled[
        sample_indices
    ],

    method="ward"

)

plt.figure(
    figsize=(12, 6)
)

dendrogram(
    linkage_matrix
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel(
    "Observations"
)

plt.ylabel(
    "Distance"
)

plt.tight_layout()

plt.savefig(
    "dendrogram.png",
    dpi=300
)

plt.show()


# ==========================================================
# DBSCAN
# ==========================================================

print("\n" + "=" * 80)
print("DBSCAN")
print("=" * 80)

dbscan = DBSCAN(

    eps=0.8,

    min_samples=8

)

dbscan_labels = (
    dbscan.fit_predict(
        X_scaled
    )
)

df["DBSCANCluster"] = (
    dbscan_labels
)

print("\nDBSCAN Cluster Counts:")

print(
    df["DBSCANCluster"]
    .value_counts()
    .sort_index()
)

# DBSCAN uses -1 for noise.

noise_count = (
    df["DBSCANCluster"]
    .eq(-1)
    .sum()
)

print(
    "\nNoise Points:",
    noise_count
)


# ==========================================================
# DBSCAN SILHOUETTE SCORE
# ==========================================================

print("\n" + "=" * 80)
print("DBSCAN EVALUATION")
print("=" * 80)

dbscan_non_noise = (
    dbscan_labels != -1
)

unique_dbscan_clusters = (
    set(
        dbscan_labels[
            dbscan_non_noise
        ]
    )
)

if len(unique_dbscan_clusters) >= 2:

    dbscan_score = (
        silhouette_score(

            X_scaled[
                dbscan_non_noise
            ],

            dbscan_labels[
                dbscan_non_noise
            ]

        )
    )

    print(
        "DBSCAN Silhouette Score:",
        dbscan_score
    )

else:

    print(
        "Silhouette Score cannot be calculated "
        "because DBSCAN produced fewer than "
        "two non-noise clusters."
    )


# ==========================================================
# DBSCAN VISUALIZATION
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(

    df["AnnualIncome"],

    df["SpendingScore"],

    c=df["DBSCANCluster"],

    alpha=0.7

)

plt.xlabel(
    "Annual Income"
)

plt.ylabel(
    "Spending Score"
)

plt.title(
    "DBSCAN Customer Clustering"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "dbscan_clusters.png",
    dpi=300
)

plt.show()


# ==========================================================
# COMPARISON
# ==========================================================

print("\n" + "=" * 80)
print("CLUSTERING ALGORITHM COMPARISON")
print("=" * 80)

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)

comparison = pd.DataFrame({

    "Algorithm": [

        "K-Means",

        "Agglomerative"

    ],

    "Silhouette_Score": [

        kmeans_score,

        hierarchical_score

    ]

})

print(
    comparison.to_string(
        index=False
    )
)


# ==========================================================
# BUSINESS INTERPRETATION
# ==========================================================

print("\n" + "=" * 80)
print("BUSINESS INTERPRETATION")
print("=" * 80)

print("""
Cluster labels such as 0, 1 and 2 do not inherently
have business meanings.

The meaning must be inferred by examining the
cluster profiles.

For example, a cluster with:

- High income
- High spending
- High purchase frequency

could potentially be interpreted as:

"High-value / loyal customers"

A cluster with:

- Lower income
- Lower spending
- Lower purchase frequency

could potentially represent:

"Low-engagement customers"

These are business interpretations, not ground-truth
labels produced by the clustering algorithm.
""")


# ==========================================================
# FINAL CLUSTER PROFILE
# ==========================================================

print("\n" + "=" * 80)
print("FINAL CUSTOMER SEGMENTS")
print("=" * 80)

final_profile = (
    df
    .groupby(
        "KMeansCluster"
    )
    .agg({

        "Age": "mean",

        "AnnualIncome": "mean",

        "SpendingScore": "mean",

        "PurchaseFrequency": "mean"

    })
    .round(2)
)

print(
    final_profile
)


# ==========================================================
# SAVE CLUSTERED DATASET
# ==========================================================

output_file = (
    "customer_clusters.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    f"\nClustered dataset saved as: "
    f"{output_file}"
)


# ==========================================================
# MINI PRACTICE
# ==========================================================

print("\n" + "=" * 80)
print("MINI PRACTICE")
print("=" * 80)

print("""
1. Try K-Means with K = 2, 3, 4, 5.

2. Plot the Elbow curve.

3. Calculate Silhouette Score for
   every K.

4. Compare Elbow and Silhouette
   recommendations.

5. Change the K-Means initialization.

6. Experiment with n_init.

7. Try Agglomerative Clustering with:

   - ward
   - complete
   - average

8. Experiment with DBSCAN:

   eps = 0.3
   eps = 0.5
   eps = 0.8
   eps = 1.0

9. Change min_samples.

10. Count DBSCAN noise points.

11. Compare cluster profiles.

12. Give business names to the clusters.

13. Determine which clustering method
    produces the most meaningful segments.

14. Explain why scaling changes clustering
    results.

15. Add another customer feature and
    observe how the clusters change.
""")


# ==========================================================
# LESSON SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("LESSON SUMMARY")
print("=" * 80)

print("""
Skills Learned

✔ Unsupervised Learning
✔ Clustering
✔ K-Means
✔ Centroids
✔ Inertia
✔ Elbow Method
✔ Silhouette Score
✔ Agglomerative Clustering
✔ Hierarchical Clustering
✔ Dendrogram
✔ DBSCAN
✔ Core Points
✔ Border Points
✔ Noise Points
✔ Feature Scaling
✔ Cluster Profiling
✔ Customer Segmentation
✔ Business Interpretation
""")

print("\nLesson 06 Completed Successfully!")
