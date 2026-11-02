import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# Cell 1: Intro
add_md("""# Gradient Descent from Scratch 📉
Gradient Descent is the optimization engine behind almost all machine learning and deep learning algorithms. 

Its goal is simple: **find the parameters (weights and biases) that minimize a loss function.**

### The Core Update Rule:
$$\\theta := \\theta - \\eta \\cdot \\nabla J(\\theta)$$

* $\\theta$: Parameter to optimize (weights $w$ or bias $b$).
* $\\eta$ (eta): Learning rate (how big of a step to take).
* $\\nabla J(\\theta)$: The gradient (slope / direction of steepest increase).
* The **minus sign** ($-$) ensures we step in the *opposite* direction of the slope (downhill!).""")

# Cell 2: Imports
add_code("""import numpy as np
import matplotlib.pyplot as plt

# Plotting settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = [8, 5]
np.random.seed(42)""")

# Cell 3: Part 1 Intro
add_md("""---
## Part 1: Gradient Descent in 1D (Visual Intuition)

Let's find the minimum of a simple 1D curve:
$$f(x) = x^2$$

Its derivative (the slope) is:
$$f'(x) = 2x$$

* If we are at $x = 4$, the slope is $f'(4) = 8$ (positive $\\implies$ curve rises to the right).
* To go downhill toward the minimum, we must step to the left:
  $$x_{\\text{new}} = x - \\eta \\cdot f'(x)$$""")

# Cell 4: 1D GD Code
add_code("""def f(x):
    return x**2

def df(x):
    return 2 * x

def gradient_descent_1d(x_start, lr, n_steps):
    history = [x_start]
    x = x_start
    for _ in range(n_steps):
        grad = df(x)
        x = x - lr * grad
        history.append(x)
    return np.array(history)

# Run Gradient Descent
x_start = 4.0
lr = 0.15
steps = 15
history = gradient_descent_1d(x_start, lr, steps)

# Plot the path down the bowl
x_vals = np.linspace(-4.5, 4.5, 200)
plt.plot(x_vals, f(x_vals), label="$f(x) = x^2$", color='navy', lw=2)
plt.plot(history, f(history), 'ro--', label=f'GD Steps (lr={lr})', markersize=6)
plt.title(f"Gradient Descent on $f(x) = x^2$ (Started at x={x_start})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

print(f"Started at: x = {x_start:.2f}")
print(f"Ended at:   x = {history[-1]:.6f} after {steps} steps (True minimum = 0.0)")""")

# Cell 5: Learning Rate Markdown
add_md("""### Why the Learning Rate ($\eta$) Matters
* **Too Small ($\eta = 0.02$):** Takes tiny baby steps; slow to converge.
* **Just Right ($\eta = 0.15$):** Smoothly slides down to the minimum.
* **Too Large ($\eta = 1.05$):** Overshoots the valley, bounces higher and higher, and **diverges**!""")

# Cell 6: Learning Rate Comparison Code
add_code("""learning_rates = [0.02, 0.15, 1.05]
titles = ["Too Small (lr = 0.02)", "Just Right (lr = 0.15)", "Too Large (lr = 1.05) - Diverges!"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
x_vals = np.linspace(-5, 5, 200)

for ax, lr_val, title in zip(axes, learning_rates, titles):
    hist = gradient_descent_1d(x_start=4.0, lr=lr_val, n_steps=10)
    ax.plot(x_vals, f(x_vals), color='gray', linestyle='--')
    ax.plot(hist, f(hist), 'ro-', markersize=5)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-1, 25)

plt.tight_layout()
plt.show()""")

# Cell 7: Part 2 Intro
add_md("""---
## Part 2: Gradient Descent for a Model (Linear Regression)

Now let's apply gradient descent to find the best **Weight ($w$)** and **Bias ($b$)** to fit a dataset.

### 1. The Model:
$$\\hat{y} = w x + b$$

### 2. The Loss Function (Mean Squared Error):
$$J(w, b) = \\frac{1}{m} \\sum_{i=1}^m (\\hat{y}^{(i)} - y^{(i)})^2$$

### 3. The Gradients (via Calculus):
$$\\frac{\\partial J}{\\partial w} = \\frac{2}{m} \\sum_{i=1}^m (\\hat{y}^{(i)} - y^{(i)}) \\cdot x^{(i)}$$
$$\\frac{\\partial J}{\\partial b} = \\frac{2}{m} \\sum_{i=1}^m (\\hat{y}^{(i)} - y^{(i)})$$""")

# Cell 8: Synthetic Data Code
add_code("""# Generate synthetic linear data: y = 3x + 4 + noise
m = 100
X = 2 * np.random.rand(m, 1)
true_w = 3.0
true_b = 4.0
y = true_w * X + true_b + np.random.randn(m, 1) * 0.5

plt.scatter(X, y, color='dodgerblue', alpha=0.7, edgecolors='k', label='Samples')
plt.title("Synthetic Training Data (Target: $y = 3x + 4$)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()""")

# Cell 9: Training Loop Code
add_code("""# Random initial weights
w = np.random.randn(1)
b = np.random.randn(1)
lr = 0.1
epochs = 60

loss_history = []
w_history = [float(w[0])]
b_history = [float(b[0])]

# Training Loop: Forward -> Loss -> Gradients -> Update
for epoch in range(epochs):
    # 1. Forward Pass (Predictions)
    y_pred = w * X + b
    
    # 2. Loss Computation (MSE)
    loss = np.mean((y_pred - y)**2)
    loss_history.append(loss)
    
    # 3. Compute Gradients
    error = y_pred - y
    grad_w = (2 / m) * np.sum(error * X)
    grad_b = (2 / m) * np.sum(error)
    
    # 4. Gradient Descent Update
    w -= lr * grad_w
    b -= lr * grad_b
    
    w_history.append(float(w[0]))
    b_history.append(float(b[0]))

print(f"Learned Weight w: {w[0]:.4f} (True: {true_w})")
print(f"Learned Bias b:   {b[0]:.4f} (True: {true_b})")
print(f"Final MSE Loss:   {loss_history[-1]:.4f}")""")

# Cell 10: Regression Results Plot
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss dropping over epochs
ax1.plot(loss_history, color='crimson', lw=2)
ax1.set_title("Loss Curve: MSE Dropping Over Time")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Mean Squared Error (MSE)")

# Plot 2: Fitted line vs actual data
ax2.scatter(X, y, color='dodgerblue', alpha=0.6, edgecolors='k', label='Data points')
x_line = np.array([[0], [2]])
ax2.plot(x_line, w * x_line + b, color='red', lw=3, label=f'GD Fit: y = {w[0]:.2f}x + {b[0]:.2f}')
ax2.plot(x_line, true_w * x_line + true_b, color='green', linestyle='--', lw=2, label='True: y = 3x + 4')
ax2.set_title("Final Fitted Line vs. Reality")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()

plt.tight_layout()
plt.show()""")

# Cell 11: Part 3 Intro
add_md("""---
## Part 3: Batch GD vs. Stochastic GD (SGD)

* **Batch Gradient Descent:** Uses all $m = 100$ samples every single step $\\implies$ smooth, direct path to minimum.
* **Stochastic Gradient Descent (SGD):** Uses **1 random sample at a time** $\\implies$ noisy and zig-zags around, but takes steps much faster and can escape shallow traps.

Let's trace both paths in **Parameter Space ($w$ vs $b$)**:""")

# Cell 12: SGD vs Batch GD Code
add_code("""# Run SGD from the same starting point
w_sgd = np.array([w_history[0]])
b_sgd = np.array([b_history[0]])
w_sgd_history = [float(w_sgd[0])]
b_sgd_history = [float(b_sgd[0])]

n_sgd_steps = 150
lr_sgd = 0.05

for step in range(n_sgd_steps):
    # Pick 1 random sample
    idx = np.random.randint(m)
    xi = X[idx:idx+1]
    yi = y[idx:idx+1]
    
    # Prediction and error for single sample
    y_pred_i = w_sgd * xi + b_sgd
    error_i = y_pred_i - yi
    
    # Gradients for this single instance
    grad_w_i = 2 * error_i * xi
    grad_b_i = 2 * error_i
    
    # Update
    w_sgd -= lr_sgd * grad_w_i.squeeze()
    b_sgd -= lr_sgd * grad_b_i.squeeze()
    
    w_sgd_history.append(float(w_sgd[0]))
    b_sgd_history.append(float(b_sgd[0]))

# Visual comparison in (w, b) space
plt.figure(figsize=(9, 6))
plt.plot(w_history, b_history, 'b-o', markersize=4, label='Batch GD (Smooth & Direct)')
plt.plot(w_sgd_history, b_sgd_history, 'r-x', markersize=3, alpha=0.5, label='SGD (Noisy & Zig-zagging)')
plt.plot(true_w, true_b, 'g*', markersize=16, label='True Minimum (w=3, b=4)')

plt.title("Path in Parameter Space ($w$ vs $b$): Batch GD vs. SGD")
plt.xlabel("Weight ($w$)")
plt.ylabel("Bias ($b$)")
plt.legend()
plt.show()""")

# Cell 13: Summary
add_md("""---
## Key Takeaways

1. **The Update Rule:** $\\theta \\leftarrow \\theta - \\eta \\cdot \\text{gradient}$. Always step against the slope to reduce loss.
2. **Learning Rate ($\eta$):** If too small, training is slow. If too large, training diverges.
3. **Batch GD vs. SGD:** 
   * **Batch GD** is stable but slow on large datasets.
   * **SGD** is fast and memory-efficient, but bounces around the minimum.
4. **Next Step (Advanced Optimizers):** Optimizers like **SGD + Momentum** and **Adam** solve the zig-zagging of SGD by adding velocity and adaptive step sizes!""")

output_path = r"c:\Users\amido\OneDrive\Desktop\lecs\ML\week2\day1\notebooks\gradient_descent.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook written to:", output_path)

