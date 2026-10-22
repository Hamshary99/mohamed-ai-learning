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
 