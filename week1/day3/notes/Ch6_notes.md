# Chapter 6
## Decision Trees

Like SVMs, Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks, and even multioutput tasks.

They ask binary (yes/no) questions and branch accordingly to reach a decision. 

> One of the many qualities of Decision Trees is that they require very little data preparation. In particular, they don’t require feature scaling or centering at all.

---

## Estimating Class Probabilities

To estimate the probability that a new instance belongs to a class:
1. It follows the questions down the tree to land in a **leaf node** (bucket).
2. It simply calculates the **percentage (fraction)** of training examples in that leaf:
   $$\text{Probability} = \frac{\text{examples of that class in the leaf}}{\text{total examples in the leaf}}$$
   *(e.g., if a leaf holds 49 Versicolor out of 54 flowers $\to 49 / 54 \approx 90.7\%$)*

---

## The CART Training Algorithm

Scikit-Learn uses the **Classification And Regression Tree (CART)** algorithm, which generates **binary trees**. 

CART algorithm is a **greedy algorithm**: it greedily searches for an optimum split at the top level, then repeats the process recursively **at each new node**.

It stops recursing once it reaches the maximum depth (defined by the max_depth hyperparameter), or if it cannot find a split that will reduce impurity.

---

## Computational Complexity

Inference/Prediction: Extremely fast—$O(\log_2(m))$, independent of feature count.
Training: Can get slow on large datasets—$O(n \times m \log_2(m))$.

---

## Gini Impurity or Entropy

Remember back in thermodynamic class, entropy was how unstable a molecule was.

In Machine Learning, it is frequently used as an impurity measure; a set’s entropy is zero when it contains instances of only one class.

As for Gini, it is another measure of impurity, not much difference from entropy, but it's computationally less expensive.

$$\text{Gini} = 1 - \sum_{i=1}^{C} p_{i}^{2}$$
$$\text{Entropy} = -\sum_{i=1}^{C} p_{i} \log_{2}(p_{i})$$

---

## Regularization Hyperparameters

One issue with Decision Trees is that they assume the training data is linear and noise free, which leads to overfitting. 

Sometimes we call it nonparametric model, because it doesn't care much about the parameters and starts assuming.

We can limit the overfitting by using regularization hyperparameters:
- Limit the maximum depth of the tree `max_depth`
- Limit the minimum number of samples required to split a node `min_samples_split`
- Limit the minimum number of samples required in a leaf node `min_samples_leaf`
- Limit the maximum number of leaf nodes `max_leaf_nodes`
- Limit the minimum number of samples required to split an internal node `min_impurity_decrease`

---

## Regression

Decision Trees are also capable of performing regression tasks. Decision Tree can't draw curves. It only knows how to ask yes/no questions and chop data into bins.

$$\text{SSE} = \sum_{i \in \text{leaf}} (y_i - \hat{y})^2$$

Instead of predicting a class label, a leaf node predicts the average target value ($\hat{y}$) of all training samples falling into that leaf.

> If you don't set a limit like max_depth, the tree will keep chopping and chopping until almost every single data point has its own private leaf.
> Its training error will be zero, but its prediction becomes an insane, jagged zigzag that memorized all the noise.

---

## Instability

The main issue with Decision Trees is that they are very sensitive to small variations in the training data.

Splits are strictly perpendicular to feature axes; dataset rotations can create unnecessarily complex boundaries.

---