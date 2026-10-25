# Chapter 7
## Ensemble Learning and Random Forests

> **Wisdom of the Crowd:** Aggregating predictions from a group of diverse predictors (**Ensemble**) almost always outperforms the single best predictor.

- **Ensemble Method:** The algorithm used to train and combine multiple models (e.g., voting or averaging).
- **Random Forest:** A classic ensemble of Decision Trees trained on random subsets of data, predicting via majority vote.
- **When to use:** Near the end of a project—after tuning a few strong individual models—to combine them for maximum performance.

---

## Voting Classifiers

- **Hard Voting:** Aggregates predictions across multiple diverse models (e.g., SVM, Logistic Regression, Decision Tree) and predicts the class with the **majority vote**.
- **Soft Voting:** Averages predicted probabilities (`predict_proba()`) across all models and picks the class with the highest average probability.
  > **Note:** Soft voting often outperforms hard voting because it gives greater weight to highly confident predictions.
- **Weak $\to$ Strong Learner:** An ensemble can become a **strong learner** (high accuracy) even if individual classifiers are **weak learners** (only slightly better than random guessing).

### The Key: Predictor Diversity & Independence
- **Independence is Essential:** Ensembles work best when predictors are as independent from one another as possible.
- **Why use different algorithms?** They make *different types of errors*, which cancel each other out during voting.
- **The Risk (Correlated Errors):** If models share the same biases or make correlated errors, they will agree on the *wrong* answer, hurting accuracy.

---

## Bagging and Pasting

Another approach is to use the **same training algorithm** for every predictor, but train each on a **different random subset** of the training set.

- **Bagging (Bootstrap Aggregating):** Sampling is performed **with replacement** (`bootstrap=True`).
- **Pasting:** Sampling is performed **without replacement** (`bootstrap=False`).

#### Sampling: Across vs. Within Predictors
- **Across predictors:** Both bagging and pasting allow different predictors to share the same training instances (Predictor 1 and Predictor 2 can both see sample A).
- **Within a single predictor:**
  - **Pasting:** A predictor sees each instance **at most once** (no duplicates in its own subset).
  - **Bagging:** A predictor can see the same instance **multiple times** (duplicates allowed).

#### Bias vs. Variance Effect
- Individual predictors have higher bias because they train on smaller subsets of data.
- However, **aggregation drastically reduces variance**.
- **Net Result:** The ensemble has a **similar bias but significantly lower variance** (generalizes much better, less overfitting).

![Decision Tree vs. Bagging Ensemble Decision Boundary](decision_tree_vs_bagging.png)



### Out-of-Bag (OOB) Evaluation

- **What are OOB Instances?**
  - In bagging, each predictor trains on only about **63% of the training instances**.
  - The remaining **~37% never seen by that predictor** are its **Out-of-Bag (OOB)** instances.

- **How it Works as a Built-in Validation Set:**
  - To evaluate any instance, aggregate predictions from **only the predictors that did not see it during training**.
  - Since those predictors never trained on it, this gives an unbiased, out-of-sample prediction.
  - Averaging this across all instances yields the overall **OOB Score**.

- **Why use it?**
  - **No separate validation set needed:** You can use 100% of your data to train the final ensemble.
  - **No cross-validation overhead:** Evaluates performance "for free" in a single training run without retraining multiple folds.

---

## Random Patches and Random Subspaces

`BaggingClassifier` can sample **features (columns)** just like it samples instances (rows), controlled by `max_features` and `bootstrap_features`.

| Method | Rows (Instances) | Columns (Features) | Key Hyperparameters | Best Used When |
| :--- | :--- | :--- | :--- | :--- |
| **Random Subspaces** | **All** instances (100%) | **Random subset** of features | `bootstrap=False, max_samples=1.0, max_features < 1.0` | High-dimensional data (e.g., images, text) |
| **Random Patches** | **Subset** of instances | **Random subset** of features | `max_samples < 1.0, max_features < 1.0` | Massive datasets (many instances AND features) |

- **Why sample features?**
  - **Breaks feature dominance:** Prevents one strong feature from being chosen at the root of every tree.
  - **Increases diversity:** Different trees look at different attributes, leading to uncorrelated errors that cancel out (slightly higher bias, much lower variance).

---

## Random Forests

An ensemble of Decision Trees trained via bagging (`bootstrap=True`, `max_samples=1.0`), optimized as `RandomForestClassifier`.

- **Convenience:** Has all hyperparameters of `DecisionTreeClassifier` (tree growth) + `BaggingClassifier` (ensemble control).
- **Extra Feature Randomness:** At each node split, it searches for the best split only within a **random subset of features** (`max_features="sqrt"` by default), instead of searching across all features.
  - **Why `splitter="random"` mimics RF:** `BaggingClassifier(DecisionTreeClassifier(splitter="random", ...))` replicates this behavior, producing almost identical predictions (~98% match).
- **Bias vs. Variance:** Individual trees are slightly less accurate (higher bias), but the forest is much more diverse, so variance drops sharply $\to$ overall superior model.

### Extra Trees


### Feature Importance

Another great feature is that Random Forests can measure the feature (column) importances. Scikit learn can calculate how a feature can reduce the impurity on average over all trees.

---

## Boosting

**Boosting** refers to any Ensemble method that **sequentially** combines several **weak learners** into a **strong learner**. Where each successive learner focuses on the mistakes of the previous ones. 

### AdaBoost (Adaptive Boosting)

One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. The relative weight of misclassified training instances is then increased


### Gradient Boosting (GBRT)

instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the **residual errors** (leftover prediction mistakes) from the previous predictor.

- **Shrinkage / Learning Rate**: Scales the contribution of each tree (`learning_rate`). Lower learning rates require more trees but produce better generalization.

- **XGBoost (Extreme Gradient Boosting)**: An optimized, ultra-fast python library implementation of Gradient Boosting with built-in regularization and automatic early stopping.

### Gradient Boosting vs AdaBoost

| Feature | Gradient Boosting (GBRT) | AdaBoost (Adaptive Boosting) |
| :--- | :--- | :--- |
| **What it trains on** | Residual errors (prediction mistakes) of previous models | Previous errors (misclassified instances) with increased weights |
| **Updating Weights** | Iterative residual fitting (no weight updates) | Adjusts weights of misclassified instances |
| **Corrections** | Addresses remaining errors by modeling them directly | Addresses errors by focusing more on misclassified data |
| **Common Algorithms** | XGBoost, LightGBM, CatBoost | AdaBoost, SAMME, SAMME.R |


---

## Stacking

Instead of using trivial function to aggregate, why not train the model to learn the best way to aggregate the predictions? Each model has its own prediction the final predictor (called a blender, or a meta learner) takes these predictions as inputs and makes the final prediction.
![alt text](image-4.png)

### Training with a Holdout Set (Blending)

1. **Split Data:** **Subset 1** trains base models; **Subset 2 (Holdout)** is kept unseen.
2. **Predict:** Base models predict on **Subset 2** (generating clean, out-of-sample predictions).
3. **Train Blender:** Base model predictions become inputs to train the blender (meta-learner).

> **Why?** Prevents overfitting. The blender learns from honest predictions on unseen data rather than memorized data.



