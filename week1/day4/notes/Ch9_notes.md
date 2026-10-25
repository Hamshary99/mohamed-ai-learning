# Chapter 9
# Unsupervised Learning: Clustering & K-Means

Unsupervised learning involves learning patterns from **unlabeled data** ($X$ without target labels $y$).

---

## 1. What is Clustering?

**Clustering** is the task of grouping similar data points together into distinct groups (**clusters**).

### Real-World Use Cases:
* **Customer Segmentation**: Grouping customers by purchasing patterns to deliver targeted promotions.
* **Data Analysis**: Visualizing high-dimensional customer or product profiles to spot natural groupings.
* **Dimensionality Reduction**: Transforming features into distances to cluster centers.
* **Semi-Supervised Learning**: Labeling just a handful of representative cluster centroids rather than thousands of raw instances.
* **Image Segmentation / Color Quantization**: Grouping pixel colors to compress images or isolate objects.

---

## 2. K-Means Clustering

K-Means is the most popular clustering algorithm. It aims to partition $m$ instances into $k$ distinct, non-overlapping clusters.

### How It Works (Lloyd's Algorithm):
1. **Initialize Centroids**: Randomly pick $k$ points to serve as initial cluster centers (**centroids**).
2. **Assign Points**: Assign every point to its closest centroid (using Euclidean distance). This partitions the space into **Voronoi cells**.
3. **Update Centroids**: Recompute each centroid as the average (mean) of all points currently assigned to it.
4. **Repeat**: Alternate steps 2 and 3 until the centroids stop moving (convergence).

---

## 3. The Core Code Walkthrough

```python
from sklearn.cluster import KMeans
import numpy as np

# 1. Fit the model
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
```

### Inspecting the Results:

#### A. Cluster Labels (`fit_predict` & `labels_`)
```python
>>> y_pred
array([4, 0, 1, ..., 2, 1, 0], dtype=int32)

>>> y_pred is kmeans.labels_
True
```
* Each number represents the cluster ID (from $0$ to $k-1$) that the sample was assigned to.
* `kmeans.labels_` stores these assignments permanently after fitting.

#### B. Cluster Centroids (`cluster_centers_`)
```python
>>> kmeans.cluster_centers_
array([[-2.80389616, 1.80117999],
       [ 0.20876306, 2.25551336],
       [-2.79290307, 2.79641063],
       [-1.46679593, 2.28585348],
       [-2.80037642, 1.30082566]])
```
* A $k \times n$ array containing the $(x_1, x_2)$ coordinates of the 5 learned cluster centers.

---

## 4. Hard Clustering vs. Soft Clustering

### Hard Clustering: `kmeans.predict(X_new)`
Assigns each new point strictly to the single nearest centroid.
```python
>>> X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
>>> kmeans.predict(X_new)
array([1, 1, 2, 2], dtype=int32)
```
* Point `[0, 2]` is closest to Centroid 1 $\to$ assigned to cluster 1.
* Point `[-3, 3]` is closest to Centroid 2 $\to$ assigned to cluster 2.

### Soft Clustering: `kmeans.transform(X_new)`
Instead of just picking the nearest cluster, `transform()` measures the **exact Euclidean distance from each point to every single centroid**:
```python
>>> kmeans.transform(X_new)
array([[2.81093633, 0.32995317, 2.9042344 , 1.49439034, 2.88633901],
       [5.80730058, 2.80290755, 5.84739223, 4.4759332 , 5.84236351],
       [1.21475352, 3.29399768, 0.29040966, 1.69136631, 1.71086031],
       [0.72581411, 3.21806371, 0.36159148, 1.54808703, 1.21567622]])
```
* **Notice row 0** (for point `[0, 2]`):
  * Distance to Centroid 0: `2.81`
  * Distance to Centroid 1: `0.33` $\leftarrow$ **Smallest distance!** That's why `predict()` chose cluster 1.
  * Distance to Centroid 2: `2.90`
  * Distance to Centroid 3: `1.49`
  * Distance to Centroid 4: `2.88`
* **Why this is useful**: It turns any 2D point into a 5-dimensional feature vector representing its proximity to the clusters (great for non-linear feature engineering).

---

## 5. Finding the Right Number of Clusters ($k$)

K-Means requires you to specify $k$ beforehand. How do we know if $k=5$ is best?

### 1. Inertia (Within-Cluster Sum of Squares)
$$\text{Inertia} = \sum_{i=1}^{m} ||x^{(i)} - \mu_{c^{(i)}}||^2$$
* Measures the sum of squared distances between each instance and its assigned centroid.
* **Lower inertia = tighter clusters**.
* Access via `kmeans.inertia_`.
* **Caution**: You cannot just pick $k$ with the lowest inertia, because inertia *always* goes down as $k$ increases (if $k=m$, inertia is 0).

### 2. The Elbow Method
* Plot **Inertia vs. $k$**.
* Look for an **"elbow"**—the bend where the curve stops dropping steeply and flattens out. That point represents the sweet spot of optimal clusters.

### 3. The Silhouette Score
A more precise metric:
$$s = \frac{b - a}{\max(a, b)}$$
* $a$: Mean distance to points in the **same cluster** (compactness).
* $b$: Mean distance to points in the **closest neighboring cluster** (separation).
* Score ranges from **$-1$ to $+1$**:
  * Near **$+1$**: Well clustered, clearly separated.
  * Near **$0$**: Close to the cluster decision boundary.
  * Near **$-1$**: Likely assigned to the wrong cluster.

---

## 6. Important Caveats & Best Practices
1. **Always scale your features**: K-Means uses Euclidean distance. If one feature is in thousands (e.g., income) and another is 0–1 (e.g., age ratio), the large feature will dominate.
2. **K-Means++ (Default)**: Scikit-Learn automatically places initial centroids far apart from each other, preventing poor local minima.
3. **Shape Assumption**: K-Means assumes clusters are spherical and roughly equal in size. It struggles with elongated shapes or clusters of varying densities.

---

## Quick Reference Summary
| Method / Attribute | What it does |
| :--- | :--- |
| `kmeans.fit_predict(X)` | Fits K-Means and returns cluster labels for each sample in $X$. |
| `kmeans.labels_` | The assigned cluster IDs ($0, 1, \dots, k-1$) for training data. |
| `kmeans.cluster_centers_` | The $(x, y)$ coordinates of each learned centroid. |
| `kmeans.predict(X_new)` | Predicts the closest cluster ID for unseen instances. |
| `kmeans.transform(X_new)` | Returns the exact distance from each point to every centroid. |
| `kmeans.inertia_` | Total within-cluster sum of squared distances (lower = tighter). |
