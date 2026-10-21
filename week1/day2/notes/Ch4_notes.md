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
*(Next up)*

## Polynomial Regression

## Learning Curves

## Regularized Linear Models

## Logistic Regression

## Exercises
