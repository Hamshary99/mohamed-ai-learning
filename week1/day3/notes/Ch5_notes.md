# Chapter 5
# Support Vector Machine(SVM)

It's a powerful ML model, that capable of figuring linear, non linear classifications, regression, even anomaly detection. Eevry ML model should have one as it works on small and medium data sets.

SVM plotting looks for the safest possible boundary between classes. Call it "Wide Street" Analogy.

Theset instances that sit on the edge of the street called support vectors. They hold up the decision boundary, nudging them shifts the whole border.

![alt text](image.png)

---

## Linear SVM Classification 

### Soft Margin vs Hard Margin:


**Hard Margin** (The perfectionist):
- Eevry single data instance must be classified correctly without crossing the border, with no exceptions. Breaks easily to messy data.
- Outliers can mess up the border.


**Soft Margin** (The pragmatist): 
Data in reality is messy and full of noises and outliers. Gotta find a sweet spot between two goals:
1. Maximize the street's width (Margin), leading to better generalization.
2. Limit margin violations, preventing too many points from stepping into the road.

> Margin violations: When instances are on the wrong side of the margin or inside the street.

![alt text](image-1.png)

---

## Non-linear SVM classification 

For non-linear data, we used to have polynomial regression to find linear relations to non-linear data. That means we gotta add high degree of polynomial features to make the data linear, but that leads to overfitting. 


> Regarding Polynomial Features, a low polynomial degree it cannot deal with very complex datasets, and with a high polynomial degree it creates a huge number of features, making the model too slow. So we resort to a trick called **Polynomial Kernel**


#### Polynomial Kernel
Based on **Mercer's Theorem**, the kernel trick calculates the exact mathematical results (dot products) as if you had added thousands of high-degree polynomial features, **without ever actually creating or storing those features in memory**. This allows SVMs to easily handle high-degree polynomial relationships with virtually no extra memory footprint.


#### Adding Similarity Features (The Concept)
Measure how close each data point is to landmark pins using a bell curve (**Gaussian RBF**):
- **Closeness Score**: $1.0$ if sitting directly on the pin; smoothly fades to $0.0$ further away.
- **The Transformation**: Replaces raw coordinates with closeness scores, lifting non-linear points into a space where they become linearly separable.
- **The Catch ($m \times m$ Memory Explosion)**:
  - If every instance ($m$) becomes a landmark, each point computes distance to all $m$ pins $\to$ creates an **$m \times m$ matrix**.
  - E.g., for $3$ points:

| Instance | Pin 1 | Pin 2 | Pin 3 |
| :--- | :---: | :---: | :---: |
| **Point 1** | **$1.0$** | $0.20$ | $0.01$ |
| **Point 2** | $0.20$ | **$1.0$** | $0.60$ |
| **Point 3** | $0.01$ | $0.60$ | **$1.0$** |

  - For $m = 100,000$, this matrix has **$10$ billion numbers ($\approx 80$ GB RAM)**, immediately crashing memory.

![alt text](image-2.png)

---

#### Gaussian RBF Kernel (The Practical SVM Magic)
`SVC(kernel="rbf")` gives the exact same result as that giant $m \times m$ matrix **without ever creating it in RAM**.

- **Hyperparameters ($\gamma$ & $C$)**:
  - **$\gamma$ (Gamma - Bell width / reach)**:
    - **High $\gamma$**: Narrow reach $\to$ wiggly boundary, wraps tight around points (**Overfitting**).
    - **Low $\gamma$**: Wide reach $\to$ smooth, simple boundary (**Underfitting**).
  - **$C$**: High $C$ = strict/narrow street; Low $C$ = relaxed/wide street.
  - *Tip*: If **overfitting**, reduce $\gamma$ and $C$. If **underfitting**, increase them.
- **Which Kernel to Pick?**:
  1. `LinearSVC` first (fastest, best for large datasets).
  2. `SVC(kernel="rbf")` next (the go-to for non-linear small/medium datasets).
  3. `SVC(kernel="poly")` only if needed.

---

### Computational Complexity: `LinearSVC` vs. `SVC`

| Class | Library | Kernel Trick? | Speed as Instances ($m$) Grow | Best Suited For |
| :--- | :--- | :---: | :--- | :--- |
| **`LinearSVC`** | `liblinear` | ❌ No | **Fast & Linear** ($10\times$ data $\approx 10\times$ time) | Large datasets (millions of rows) |
| **`SVC`** | `libsvm` | ✅ Yes | **Dreadfully slow** ($10\times$ data $\approx 100\times$ to $1000\times$ time!) | Complex small-to-medium datasets |

- **Why does `SVC` get dreadfully slow on large datasets? (The "Party Handshakes" Analogy)**:
  - The kernel trick works by comparing **pairs of instances**:
    - $10$ people in a room shaking hands with each other $= 45$ handshakes.
    - $100$ people $= 4,950$ handshakes.
    - $100,000$ people $= \mathbf{5 \text{ billion pairwise comparisons}}$ ($O(m^2)$ to $O(m^3)$)!
  - Every time you double the rows, the workload **quadruples or octuples**.
- **The Silver Lining**: `SVC` handles datasets with **huge numbers of features** ($n$) or sparse data (like word counts in text) extremely well, as long as the row count ($m$) isn't gigantic.



---

## SVM Regression

SVM not only accepting linear and non-linear classifications but it can be used for linear and non-linear regression.

Unlike sVM classification which tries to find a margin between classes and minimize margin violations, SVM regression tries to find a margin around the prediction line and minimize margin violations.

The width of the street is controlled by a hyperparameter $\epsilon$.![alt text](image-3.png)

