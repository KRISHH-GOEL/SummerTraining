# Lesson 06 – Unsupervised Learning & Clustering

## 📌 Objective

The objective of this lesson was to understand Unsupervised Machine Learning and implement clustering algorithms.

Unlike supervised learning, unsupervised learning works with datasets where the target variable is not provided.

The model attempts to discover hidden structures, patterns, groups, or relationships within the data.

---

# 📚 Topics Covered

## Unsupervised Learning

- What is Unsupervised Learning?
- Supervised vs Unsupervised Learning
- Clustering
- Cluster
- Centroid
- Distance-based grouping
- Intra-cluster similarity
- Inter-cluster separation

## K-Means Clustering

- K-Means intuition
- Centroids
- Assignment step
- Update step
- Iterative optimization
- `KMeans()`
- `n_clusters`
- `init`
- `n_init`
- `max_iter`
- Random initialization

## Choosing the Number of Clusters

- Elbow Method
- Within-Cluster Sum of Squares
- Inertia
- Silhouette Score

## Hierarchical Clustering

- Hierarchical clustering
- Agglomerative clustering
- Bottom-up approach
- Dendrogram
- Linkage methods
- `AgglomerativeClustering()`

## DBSCAN

- Density-based clustering
- Core points
- Border points
- Noise points
- `eps`
- `min_samples`
- Advantages over K-Means

## Cluster Evaluation

- Inertia
- Silhouette Score
- Cluster visualization
- Cluster size
- Business interpretation

## Feature Scaling for Clustering

- Why scaling matters
- StandardScaler
- Distance-based algorithms
- Effect of feature magnitude

---

# 🧠 What Is Unsupervised Learning?

In supervised learning, the dataset contains:

```text
Features + Target
```

Example:

```text
Age   Income   Churn
25    30000      0
45    80000      1
```

In unsupervised learning:

```text
Features only
```

Example:

```text
Age   Income
25    30000
45    80000
32    45000
```

There is no predefined target.

The algorithm attempts to discover patterns within the data.

---

# 🔵 What Is Clustering?

Clustering is the process of grouping similar observations together.

Example:

```text
Customer Data
      ↓
Clustering Algorithm
      ↓
┌─────────┬─────────┬─────────┐
│Cluster 0│Cluster 1│Cluster 2│
└─────────┴─────────┴─────────┘
```

Observations within the same cluster should ideally be relatively similar.

Different clusters should ideally be well separated.

---

# 🔄 K-Means Clustering

K-Means divides observations into `K` clusters.

The algorithm works approximately as follows:

```text
Choose K
   ↓
Initialize Centroids
   ↓
Assign Points to Nearest Centroid
   ↓
Recalculate Centroids
   ↓
Repeat
   ↓
Stop when assignments stabilize
```

The objective is to minimize the within-cluster sum of squared distances.

---

# 📐 Inertia

Inertia measures the total squared distance between observations and their assigned cluster centroid.

Conceptually:

```text
Inertia =
Σ distance(point, assigned centroid)²
```

Lower inertia generally means tighter clusters.

However:

> Inertia always tends to decrease as the number of clusters increases.

Therefore, inertia should not be used alone to select `K`.

---

# 📉 Elbow Method

The Elbow Method calculates inertia for different values of `K`.

Example:

```text
K       Inertia

2       5000
3       3000
4       1800
5       1500
6       1400
7       1350
```

The point where the improvement starts becoming considerably smaller can be considered a candidate for `K`.

This point is called the:

```text
Elbow
```

The elbow method is a heuristic, not an absolute rule.

---

# 📊 Silhouette Score

The Silhouette Score evaluates how well observations fit within their assigned clusters.

The score ranges approximately from:

```text
-1 → 1
```

Interpretation:

```text
Higher positive value
→ Better-defined clustering

Around 0
→ Overlapping clusters

Negative value
→ Possible incorrect assignment
```

A higher score is generally preferable, but the final choice should also consider domain meaning.

---

# 🌳 Hierarchical Clustering

Hierarchical clustering creates a hierarchy of clusters.

Agglomerative clustering follows a bottom-up approach.

Initially:

```text
Every point = Individual cluster
```

Then:

```text
Closest clusters
       ↓
Merge
       ↓
Closest clusters
       ↓
Merge
       ↓
Continue...
```

The result can be visualized using a dendrogram.

---

# 🌲 Dendrogram

A dendrogram represents hierarchical relationships between observations.

Conceptually:

```text
        ┌───────────────┐
        │               │
     ┌──┴──┐         ┌──┴──┐
     │     │         │     │
     A     B         C     D
```

The height at which clusters merge represents their dissimilarity according to the chosen linkage/distance definition.

---

# 🌐 DBSCAN

DBSCAN stands for:

```text
Density-Based Spatial Clustering
of Applications with Noise
```

Instead of requiring the number of clusters beforehand, DBSCAN identifies dense regions.

It identifies:

### Core Points

Points having enough neighboring observations within the specified radius.

### Border Points

Points near a core region but not dense enough to be core points themselves.

### Noise Points

Points that do not belong to any sufficiently dense region.

---

# ⚙️ DBSCAN Parameters

## `eps`

Maximum neighborhood radius used to determine neighboring points.

## `min_samples`

Minimum number of observations required to form a dense neighborhood.

Choosing these values appropriately is important.

---

# 🆚 K-Means vs Hierarchical vs DBSCAN

| Algorithm | Need K? | Handles Noise? | Main Idea |
|---|---:|---:|---|
| K-Means | Yes | No | Centroid-based |
| Hierarchical | No* | Limited | Hierarchy |
| DBSCAN | No | Yes | Density-based |

`*` Agglomerative clustering can be configured with a desired number of clusters, while hierarchical analysis can also be explored through a dendrogram.

---

# ⚠️ Why Scaling Matters

Many clustering algorithms rely on distances.

Suppose:

```text
Age:    20 – 70
Income: 20,000 – 200,000
```

Income has a much larger numerical scale.

Without scaling, income could dominate distance calculations.

Therefore, StandardScaler is commonly applied before distance-based clustering.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 🔄 Clustering Workflow

```text
Dataset
   ↓
Select Relevant Features
   ↓
EDA
   ↓
Handle Missing Values / Outliers
   ↓
Scale Features
   ↓
Choose Clustering Algorithm
   ↓
Determine K / Parameters
   ↓
Train Clustering Model
   ↓
Assign Cluster Labels
   ↓
Evaluate Clusters
   ↓
Visualize
   ↓
Interpret Clusters
   ↓
Derive Business Insights
```

---

# 📏 Important Evaluation Methods

### Inertia

Used primarily with K-Means.

Lower is generally better, but it must be interpreted relative to the number of clusters.

### Silhouette Score

Measures cluster cohesion and separation.

Higher is generally better.

### Visualization

Visual inspection can help identify:

- Cluster overlap
- Outliers
- Separation
- Cluster density

---

# 🧪 Dataset Used

The implementation uses a synthetic customer segmentation dataset.

Features include:

- Age
- Annual Income
- Spending Score
- Purchase Frequency

The objective is to identify groups of customers with similar characteristics.

Potential business interpretations might include:

```text
High-value customers
Budget-conscious customers
Frequent buyers
Low-engagement customers
```

These labels are interpretations applied after examining the cluster characteristics; they are not learned target labels.

---

# 🎯 Business Applications

Clustering is widely used for:

- Customer segmentation
- Market segmentation
- Product grouping
- Recommendation systems
- Anomaly detection
- Geographic segmentation
- Document clustering
- Image segmentation
- Behaviour analysis

---

# ⚠️ Important Limitations

Clustering does not automatically produce meaningful business segments.

A mathematically valid cluster may not have practical significance.

Always inspect:

- Feature selection
- Scaling
- Number of clusters
- Cluster sizes
- Cluster characteristics
- Stability
- Domain/business meaning

---

# 📝 Mini Practice

1. Run K-Means with `K = 2`.

2. Try values from `K = 2` to `K = 10`.

3. Plot the Elbow curve.

4. Calculate the Silhouette Score for different values of `K`.

5. Select a suitable `K`.

6. Visualize the resulting clusters.

7. Calculate cluster sizes.

8. Calculate average feature values for every cluster.

9. Give meaningful business descriptions to the clusters.

10. Try Agglomerative Clustering.

11. Experiment with different linkage methods.

12. Try DBSCAN.

13. Experiment with different `eps` and `min_samples`.

14. Identify DBSCAN noise points.

15. Compare K-Means, Agglomerative Clustering and DBSCAN.

---

# 🎯 Key Takeaway

Clustering is not simply:

```text
Run algorithm → Get clusters
```

A proper clustering workflow is:

```text
Understand Data
      ↓
Select Features
      ↓
Scale Data
      ↓
Try Multiple Approaches
      ↓
Evaluate Clusters
      ↓
Visualize
      ↓
Interpret
      ↓
Validate Business Meaning
```

The most important skill is not just implementing clustering algorithms, but understanding whether the resulting clusters actually represent useful patterns.
