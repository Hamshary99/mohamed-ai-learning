# Chapter 4 — Training Models

---

## 1. Linear Regression

### 💡 What Should I Take Away From This?
- **The Model**: A linear model makes a prediction $\hat{y}$ by calculating a weighted sum of the input features, plus a bias constant (intercept):
  $$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \mathbf{\theta}^T \mathbf{x}$$
  - $\theta_0$: Bias term (where the line starts on the y-axis).
  - $\theta_1, \dots, \theta_n$: Feature weights (the slope / impact of each feature).
- **The $x_0 = 1$ Trick**: We always add a dummy feature column of all $1$s (`X_b = np.c_[np.ones(...), X]`) so that the bias term $\theta_0$ can be computed inside the exact same dot product vector multiplication ($\mathbf{\theta}^T \mathbf{x}$).
- **The Performance Scoreboard (Cost Function)**: We use **Mean Squared Error (MSE)**. Training a model means finding the parameter vector $\mathbf{\theta}$ that minimizes this MSE.
- **Why MSE is Well-Behaved**: The MSE cost function for Linear Regression is a smooth, bowl-shaped convex function. That means it has only **one global minimum**—there are no local traps.

---

### 1.1 The Normal Equation

$$\hat{\mathbf{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 💡 What Should I Take Away From This?
- **What it is**: It is an analytical "closed-form" solution. By setting the calculus derivative of MSE to zero, this exact formula gives you the optimal weights in **one single step**—no iterative loops, no learning rate to tune.
- **How Scikit-Learn Does It (`LinearRegression`)**:
  - Scikit-Learn does **not** use the Normal Equation directly.
  - Instead, it computes the **SVD pseudoinverse** ($\hat{\mathbf{\theta}} = \mathbf{X}^+ \mathbf{y}$ using `scipy.linalg.lstsq`).
  - *Why?* If features are redundant/collinear (e.g., length in feet and meters) or if you have more features than samples ($n > m$), the matrix $(\mathbf{X}^T \mathbf{X})$ cannot be inverted and the Normal Equation breaks. But the SVD pseudoinverse **always works**.

---

### 1.2 Computational Complexity

### 💡 What Should I Take Away From This?
- **Scaling with Features ($n$)**:
  - Inverting an $(n+1) \times (n+1)$ matrix in the Normal Equation scales as roughly $\mathcal{O}(n^{2.4})$ to $\mathcal{O}(n^3)$.
  - Scikit-Learn’s SVD pseudoinverse scales at roughly $\mathcal{O}(n^2)$.
  - **The Takeaway**: Both methods become painfully slow when you have many features ($n > 10,000$).
- **Scaling with Training Instances ($m$)**:
  - Both methods scale linearly ($\mathcal{O}(m)$) with instances, but **they require loading the entire dataset into RAM at once**.
  - **The Limitation**: If your dataset has millions of instances that exceed your computer's memory, neither closed-form method can be used.
- **Why this leads to Gradient Descent (Section 2)**:
  - When you have huge datasets or hundreds of thousands of features, we leave direct formulas behind and use **Gradient Descent**, which trains step-by-step and scales easily.

---

## 2. Gradient Descent *(Next Up)*
### 2.1 Batch Gradient Descent
### 2.2 Stochastic Gradient Descent
### 2.3 Mini-batch Gradient Descent

---

## 3. Polynomial Regression *(Upcoming)*

---

## 4. Learning Curves *(Upcoming)*
### 4.1 The Bias/Variance Trade-Off

---

## 5. Regularized Linear Models *(Upcoming)*
### 5.1 Ridge Regression
### 5.2 Lasso Regression
### 5.3 Elastic Net
### 5.4 Early Stopping

---

## 6. Logistic Regression *(Upcoming)*
### 6.1 Estimating Probabilities
### 6.2 Training and Cost Function
### 6.3 Decision Boundaries
### 6.4 Softmax Regression
