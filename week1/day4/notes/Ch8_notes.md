# Chapter 8
# Dimensionality Reduction

In real life, features can be too much to the poin that it slows down and harms the performance per instance, which could result in serious cases of overfitting. 

**The Goal**: Reduce the number of features without losing key information, turning an intractable problem into a fast, solvable one, or reducing dimensions to 2D/3D for **Data Visualization**.


## Main Approaches for Dimensionality Reduction

### Projection

Projecting data to a lower-dimensional space while trying to preserve as much information as possible. By assuming data lies in a lower dimensional flat plane. 
It's like trying to turn a 3D object squashing it to 2D on a paper. 

But it might fail for 3D data that can twist or fold like a swiss roll. 


### Manifold Learning

Manifold is basically a lower dimensional space than the original data space. 
Like how earth is flat 2D surface while it is 3D globe in reality. 

Manifold Learning: Models the curved $d$-dimensional surface (manifold) on which data lies and "unrolls" it instead of squashing it into 2D manifold.
It focuses on local neighborhoods: if point $A$ and point $B$ are close along the surface of the sheet, keep them close; if they are on different spirals of the roll, keep them apart.

**The Manifold Assumption**: Real-world high-dimensional data (like handwritten digits) usually lies close to a much lower-dimensional manifold.

---

## Principal Component Analysis (PCA)

### Preserving Variance

When projecting higher-dimensional data onto a lower-dimensional hyperplane (e.g., 2D to 1D), we want to choose the axis that preserves the **maximum amount of variance**.

* **Variance = Information**: The wider the points are spread out along the chosen axis, the less information is lost.
* In **Figure 8-7**:
  * **Solid line ($c_1$)**: Aligns with the longest stretch of data $\to$ preserves maximum variance (points stay spread between -1.5 and 1.5). This becomes the **1st Principal Component**.
  * **Dotted line ($c_2$)**: Orthogonal to the spread $\to$ destroys variance (squashes all points into a tight clump near 0).

![alt text](image-1.png)


* **Key Insight**: Choosing the axis of **maximum variance** also **minimizes the projection distance** (mean squared distance between original points and projected points).


### Principal Components

### Explained Variance Ratio

### Choosing the Number of Dimensions

### PCA for compression

### Randomized PCA

### Incremental PCA (IPCA)

---

## Kernel PCA

---

## LLE

---

## Other dimensionality reduction techniques

