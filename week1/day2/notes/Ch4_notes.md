# Chapter 4. Training Models

## Linear Regression
- **Equation**: $\hat{y} = \theta_0 + \theta_1 x_1 + \dots = \mathbf{\theta}^T \mathbf{x}$
  - Add $x_0 = 1$ column (`np.c_[np.ones(...), X]`) to include the bias term in the dot product.
- **Cost Function**: Mean Squared Error (MSE). A convex bowl with only one global minimum.

### The Normal Equation
- **Formula**: $\hat{\mathbf{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ (calculates optimal weights in 1 step).
- **Scikit-Learn (`LinearRegression`)**: Uses SVD pseudoinverse ($\hat{\mathbf{\theta}} = \mathbf{X}^+ \mathbf{y}$) instead. Faster ($\mathcal{O}(n^2)$ vs $\mathcal{O}(n^3)$) and never fails if features are collinear.

### Computational Complexity
- **Features ($n$)**: Slow for $n > 10,000$ due to matrix computation.
- **Instances ($m$)**: Linear, but requires all data to fit in RAM.
- **Why Gradient Descent?** Handles millions of features and out-of-core streaming data that doesn't fit in RAM.

---

## Gradient Descent

- **Goal**: Start with random weights $\mathbf{\theta}$ and gradually tweak them downhill until error (MSE) is minimized.
- **The Learning Rate ($\eta$)**: The key hyperparameter controlling **step size**.
  - **Too small**: Needs way too many iterations to reach the bottom.
  - **Too large**: Jumps across the minimum, bounces around, or diverges.
- **Feature Scaling**: Always scale features (e.g., `StandardScaler`). On unscaled data, gradient descent walks in slow, inefficient zigzags.

---

### Batch Gradient Descent

- **Partial Derivative**: Asks: *"If I nudge $\theta_j$ a tiny bit, does the error go up or down, and by how much?"*
- **Gradient Vector ($\nabla$)**: A vector packing all partial derivatives ($\theta_0, \theta_1, \dots$) together. Points **uphill** toward higher error.
- **The Step Formula**:
  $$\mathbf{\theta}^{(\text{next step})} = \mathbf{\theta} - \eta \nabla_{\mathbf{\theta}}\text{MSE}(\mathbf{\theta})$$
  - We **subtract** because we want to move away from the top of the hill (**downhill** toward minimum error).
- **Trade-off**:
  - 👍 Scales very well with many features ($n$).
  - 👎 Painfully slow on huge datasets ($m$) because it scans the **entire batch** of data for every single step.

> To find a good learning rate, you can use grid search

### Stochastic Gradient Descent

### Mini-batch Gradient Descent





## Polynomial Regression

## Learning Curves

## Regularized Linear Models

## Logistic Regression

## Exercises
