# Chapter 4 — Training Models Notes

## 1. Linear Regression — Intuition & Formulation

### What is Linear Regression?
Linear Regression predicts a continuous target $\hat{y}$ by computing a **weighted sum** of the input features, plus a constant called the **bias (or intercept)** term.

For a single instance with $n$ features:
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

- $\hat{y}$: The predicted value.
- $n$: Number of features.
- $x_i$: The $i^{\text{th}}$ feature value.
- $\theta_0$: The **bias term** (where the line crosses the y-axis when all features are 0).
- $\theta_1, \theta_2, \dots, \theta_n$: The **feature weights** (the slope for each feature).

### Vectorized Form
To make math and code efficient, we express this as a dot product:
$$\hat{y} = h_{\mathbf{\theta}}(\mathbf{x}) = \mathbf{\theta}^T \mathbf{x}$$

- $\mathbf{\theta}$: Parameter vector $\begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix}$.
- $\mathbf{x}$: Feature vector $\begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix}$, where **$x_0 = 1$ is always added as a dummy bias feature** (so that $\theta_0 \times 1 = \theta_0$).

---

### How Do We Measure "Goodness"? (The Cost Function)
To train the model, we want parameters $\mathbf{\theta}$ that minimize how far our predictions are from actual values. We use the **Mean Squared Error (MSE)**:

$$\text{MSE}(\mathbf{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \mathbf{\theta}^T \mathbf{x}^{(i)} - y^{(i)} \right)^2$$

- $m$: Number of training instances.
- $\mathbf{x}^{(i)}, y^{(i)}$: The $i^{\text{th}}$ training sample and its true target.

---

## 2. Solving Linear Regression: Closed-Form (Analytical) Solutions

> **What does "Closed-Form" mean?**  
> It is an exact mathematical equation that calculates the optimal $\mathbf{\theta}$ directly in **one step**—no iterative loops, no guessing, no learning rate.

There are two primary closed-form approaches:

### Approach 1: The Normal Equation
Setting the calculus derivative (gradient) of MSE to 0 gives the exact analytical formula:

$$\hat{\mathbf{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

- $\mathbf{X}$: The feature matrix of shape $(m, n+1)$, including the extra column of 1s for $x_0$.
- $\mathbf{y}$: The vector of target values of shape $(m, 1)$.

```python
import numpy as np

# Add x0 = 1 column to X
X_b = np.c_[np.ones((100, 1)), X]

# Compute optimal theta directly
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

#### Limitation of the Normal Equation:
- **Computational Complexity**: Inverting the $(n+1) \times (n+1)$ matrix $(\mathbf{X}^T \mathbf{X})^{-1}$ typically takes $\mathcal{O}(n^{2.4})$ to $\mathcal{O}(n^3)$. 
  - If you double the features, computation time increases by roughly $2^3 = 8\times$.
  - It becomes very slow when $n > 10,000$ features.
- **Singular (Non-Invertible) Matrices**: If features are redundant (e.g., collinear) or if $m < n$ (more features than instances), $(\mathbf{X}^T \mathbf{X})$ cannot be inverted.

---

### Approach 2: SVD Pseudoinverse (Scikit-Learn's Method)
Scikit-Learn's `LinearRegression` does **not** invert $(\mathbf{X}^T \mathbf{X})$. Instead, it computes the **Moore-Penrose pseudoinverse** ($\mathbf{X}^+$) using **Singular Value Decomposition (SVD)**:

$$\hat{\mathbf{\theta}} = \mathbf{X}^+ \mathbf{y}$$

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Intercept (theta_0):", lin_reg.intercept_)
print("Slope (theta_1):", lin_reg.coef_)
```

#### Why SVD Pseudoinverse is Better than the Normal Equation:
1. **Always Defined**: Even if $(\mathbf{X}^T \mathbf{X})$ is non-invertible or singular, $\mathbf{X}^+$ can always be computed.
2. **More Efficient & Numerically Stable**: Scales roughly as $\mathcal{O}(n^2)$ instead of $\mathcal{O}(n^3)$, and avoids floating-point precision issues with near-singular matrices.

---

### 💡 Computational Summary: When to Use What?

| Method | Instances ($m$) Scaling | Features ($n$) Scaling | Out-of-Core / Streaming? | Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Equation** | Fast: $\mathcal{O}(m)$ | Slow: $\mathcal{O}(n^3)$ | No (needs full data in RAM) | None |
| **SVD Pseudoinverse (`LinearRegression`)** | Fast: $\mathcal{O}(m)$ | Moderate: $\approx \mathcal{O}(n^2)$ | No (needs full data in RAM) | None |
| **Gradient Descent** *(Next Section)* | Fast for large datasets | Fast for millions of features! | Yes (Stochastic / Mini-batch) | Learning rate ($\eta$), epochs |
