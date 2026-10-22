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
- It picks random points and calculates the gradient based on that single instance.
- It's very fast.
- It's noisy and can bounce around the minimum, relying on the average to make it through.
- It's good for large datasets that don't fit in RAM.

One workaround is reducing the learning rate

### Mini-batch Gradient Descent
- Computes gradients on **small random sets of instances** (mini-batches, typically 20–50 samples).
- **GPU / Hardware boost**: Matrix operations over small batches are highly optimized on hardware.
- **Path**: Less erratic/jumpy than **Stochastic GD**, but walks around near the bottom instead of stopping exactly at the minimum like Batch GD.
- **Local Minima note**: For Linear Regression (MSE), there are *no* local minima (it's a convex bowl). For complex models (like neural nets), Mini-batch may have a slightly harder time escaping local minima than wild SGD, but settles closer to the bottom.

---

### Comparison of Linear Regression Algorithms (Table 4-1 from the book)

| Algorithm | Large $m$ (Samples) | Out-of-core (Disk/Stream) | Large $n$ (Features) | Hyperparameters | Feature Scaling Required? | Scikit-Learn Class |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal Equation** | Fast | No | Slow ($\mathcal{O}(n^3)$) | 0 | No | N/A |
| **SVD** | Fast | No | Slow ($\mathcal{O}(n^2)$) | 0 | No | `LinearRegression` |
| **Batch GD** | Slow | No | Fast | 2 ($\eta$, iterations) | **Yes** | `SGDRegressor` |
| **Stochastic GD** | Fast | **Yes** | Fast | $\ge 2$ (schedule) | **Yes** | `SGDRegressor` |
| **Mini-batch GD** | Fast | **Yes** | Fast | $\ge 2$ (+ batch size) | **Yes** | `SGDRegressor` (`partial_fit`) |

---

## Polynomial Regression

Even if the data isn't a simple straight line, we can use linear regression to model a linear relation out of non linear one.

Though some polynomial regression can lead to great degree of overfitting.

---

## Learning Curves

We used cross-validation to get an estimate of a model’s generalization performance.

Another way it to look at the plotted learning curve for both training and validation set.

> Bias
This part of the generalization error is due to wrong assumptions, such as assuming
that the data is linear when it is actually quadratic.

> Variance
This part is due to the model’s excessive sensitivity to small variations in the
training data.

> Irreducible error
This part is due to the noisiness of the data itself.

---

## Regularized Linear Models

Regularization combats overfitting, so for example when polynomial regression overfits the data, we reduce the degree of the polynomial.

---  
### Ridge Regression
- A linear model that adds a penalty term to the cost function to reduce the magnitude of the weights. Thus regulazing the linear model.
- **Cost Function**: $J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2$

### Lasso Regression
- **Penalty**: Uses the $\ell_1$ norm (absolute values) instead of $\ell_2$:
  $$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + \alpha \sum_{i=1}^{n} |\theta_i|$$
- **Automatic Feature Selection**: Drives unimportant feature weights completely to **0**, creating a **sparse model**.
- **Scikit-Learn**: `from sklearn.linear_model import Lasso` (`Lasso(alpha=0.1)`).

### Elastic Net
- It's the middle ground between Ridge and Lasso Regression. Mixing both regularization terms.

$$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + r_1 \sum_{i=1}^{n} |\theta_i| + r_2 \sum_{i=1}^{n} \theta_i^2$$

> Ridge is a good default, but if you suspect that only a few features are actually useful, you should prefer Lasso or Elastic Net since they tend to reduce the useless features’ weights down to zero.

### Early Stopping
- A "free lunch" regularization technique for iterative learning algorithms (like Gradient Descent).
- **How it works**: Monitor the validation error during training and **stop training as soon as the validation error reaches a minimum** (before overfitting starts pushing it back up).
- **Tip (Noisy Curves)**: With Stochastic/Mini-batch GD, wait until the validation error hasn't improved for a set number of rounds (patience), then roll back the model weights to the minimum point.

---

## Logistic Regression

It's estimating if the instances belongs to a said classification. Exceeding 50% then it counts it as belonging to the said classification.

### Decision Boundaries 
- **The "Line in the Sand"**: The exact threshold where the model is having a 50/50 existential crisis ($P = 0.5$).
  - One side ($P > 0.5$): Classified as Class 1.
  - Other side ($P < 0.5$): Classified as Class 0.
- **Dimensionality**:
  - 1 feature $\to$ A single cutoff point (e.g., petal width $= 1.66$ cm).
  - 2 features $\to$ A straight line dividing the 2D plane.
  - Multiclass (Softmax) $\to$ Linear borders slicing space between classes.
- **The 2D Decision Boundary Equation** (Why that line formula works):
  - Model predicts probability $P = \sigma(\theta_0 + \theta_1 x_0 + \theta_2 x_1)$.
  - The 50/50 boundary is where $P = 0.5$, which happens when $\theta_0 + \theta_1 x_0 + \theta_2 x_1 = 0$.
  - Solving for $x_1$ (y-axis, Petal width) gives the straight line equation:
    $$x_1 = -\frac{\theta_1 x_0 + \theta_0}{\theta_2} = -\frac{\text{coef}[0][0] \cdot x_0 + \text{intercept}[0]}{\text{coef}[0][1]}$$
- *Fun rule*: Think of it like a strict club bouncer: $1.661$ cm? *"Welcome to the Virginica VIP lounge!"* $1.659$ cm? *"Sorry buddy, you're Not Virginica tonight."*

### Softmax Regression (Multinomial Logistic Regression)

The logistic regression can support classifying to several classes per instance. Simply by taking an instamce x and scoring how close that instance to the available classes.

Just like the Logistic Regression classifier, the Softmax Regression classifier predicts
the class with the highest estimated probability.

> Softmax is multiclass not multi-output so it takes the highest class's score. 



---
