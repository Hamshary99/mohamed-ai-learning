# Chapter 2 — End-to-End Machine Learning Project

## 1. Working with Real Data & Data Pipelines

What's more important than just building the model is figuring out the objective of the model and what it needs to do, and what algorithms and performance measures we use to achieve that objective.

### Data Pipelines
- **Definition:** A sequence of data processing components that each process data and pass the output to the next component through a data store (database or files).
- **How it works:** Each step reads data, processes it, and writes the result to a data store. The next step reads from there later. The data store is the only interface between pipeline steps.
- **Data store:** The storage middleman where pipeline results are saved for downstream components to pick up.

#### Advantages:
- Each step is simple, modular, and self-contained.
- Different teams can develop and own different steps without conflicting with each other.
- If one step fails, downstream steps can temporarily continue running on the last saved output.

#### Disadvantage:
- A broken step can go unnoticed for a while, causing data to become stale and model performance to silently degrade without triggering a loud error.

#### Fix:
- Proactive monitoring and automated alerts on data freshness and output distribution.

#### Notes:
- In decoupled pipelines, steps pass data through independent intermediate storage. If an upstream step crashes, it doesn't harm the downstream. Without active monitoring, the down stream works with stale data.

---

## 2. Performance Measures: RMSE vs. MAE

Both metrics quantify how far predictions $\hat{y}$ deviate from true target values $y$ (lower is always better).

- **RMSE (Root Mean Square Error):** Squares the errors, averages them, and takes the square root. Large errors are penalized much more heavily because errors are squared before averaging. Measured in the same units as the target variable.
- **MAE (Mean Absolute Error):** The average of the absolute differences $|y - \hat{y}|$. Treats all errors the same.

> 📝 **Side Note / Correction (MAE Definition):**
> In the original notes, MAE was labeled *(Mean Average Error)*. The correct term is **Mean Absolute Error** (the "A" stands for **Absolute** values $|y - \hat{y}|$; "Mean Average" would be redundant since mean and average are synonyms).

### When to Use Which:
- Use **RMSE by default** (preferred for regression when errors are roughly bell-shaped / normally distributed).
- Use **MAE** when the dataset contains **many significant outliers** that would distort an RMSE calculation.
- *Example:* An RMSE of $50k on house prices indicates a typical error of roughly $50k.

---

## 3. Creating a Test Set

- **The 20% Rule:** Set aside ~20% of the data before exploring it, and do not inspect it.
- **Reason (Data Snooping Bias):** If you let the test set influence your choice of features, transformations, or model architectures, your performance estimates will be overly optimistic, leading to poor generalization in production (overfitting).

### Sampling Challenges:
- A purely random 20% split is fine for massive datasets, but smaller datasets suffer from **sampling bias**.
- If a population consists of 51.3% females and 48.7% males, a representative survey of 1,000 individuals must mirror those exact proportions (513 females, 487 males). Pure randomness risks an unrepresentative split that skews model training.

### Stratified Sampling:
- **Definition:** Divide the population into homogeneous subgroups (**strata**), then sample from each stratum in proportion to its representation in the full dataset.
- Guarantees that the test set accurately mirrors the overall population across critical attributes (e.g. `median_income` categories).

> 📝 **Side Note / Notebook Connection (Stable Splitting):**
> In [scratch_ch2.ipynb](file:///c:/Users/amido/OneDrive/Desktop/lecs/ML/week1/day1/notebooks/scratch_ch2.ipynb), we saw that `np.random.seed(42)` keeps random splits consistent on static data, but breaks whenever new data is fetched.
> A **hash-based identifier** (`crc32` on row index or geographic coordinates `longitude * 1000 + latitude`) ensures the test set stays stable across dataset updates without data leakage.
> np.random(42) keeps data shuffling consistent only if the db never changes. Once adding new data it shifts indexes and ruins the split. In that sense, hash-based splitting is superior. It's a bit more expensive computationally but worth it in the long run.

---

## 4. Discover & Visualize Data: Correlations

- The standard correlation coefficient (Pearson's $r$) ranges from **–1 to +1**:
  - **+1**: Strong positive linear correlation (as $x$ increases, $y$ increases).
  - **–1**: Strong negative linear correlation (as $x$ increases, $y$ decreases; e.g. prices slightly decrease moving north).
  - **0**: No linear correlation.

> 📝 **Side Note / Correction (Correlation = 0):**
> A correlation coefficient of $0$ does **not** mean there is "no relation at all" — it only means there is **no linear relationship**.
> Two variables can have a strict non-linear relationship (e.g. $y = x^2$ over $[-1, 1]$), yet their Pearson correlation $r$ will be exactly $0$.

- **Tools:** Use Pandas `scatter_matrix` or `housing.corr(numeric_only=True)` to discover linear patterns between attributes.
- **Feature Engineering:** Always experiment with attribute combinations (ratios like `rooms_per_household`, `bedrooms_per_room`, `population_per_household`) before training models. A ratio feature often has a significantly stronger correlation with the target than raw total counts.

---

## 5. Prepare the Data for Machine Learning Algorithms

### 5.1 Data Cleaning (Handling Missing Values / NaNs)
Machine Learning algorithms generally cannot handle missing values (`NaN`). Three common strategies:
1. **Option 1 (Drop rows):** Remove instances missing the value (`housing.dropna(subset=["total_bedrooms"])`).
2. **Option 2 (Drop attribute):** Remove the entire column (`housing.drop("total_bedrooms", axis=1)`).
3. **Option 3 (Imputation):** Fill missing entries with a summary statistic (median, mean, or zero).

- **Best Practice:** Use Scikit-Learn's `SimpleImputer(strategy="median")`.
- **Golden Rule (No Data Leakage):** Always compute the median on the **training set only** and store it via `imputer.statistics_`. Use those exact frozen statistics to transform validation data, test data, and production data.

---

### 5.2 Handling Text and Categorical Attributes
Machine Learning models require numerical inputs, so categorical text must be converted:
1. **Ordinal Encoding (`OrdinalEncoder`):** Maps ordered categories to integer ranks (e.g. `bad=0, mid=1, good=2, best=3`).

> 📝 **Side Note / Correction (Ordinal vs. Ordinary):**
> In the original notes, this was written as *(Ordinary encoding)*. The formal ML term is **Ordinal Encoding** (`OrdinalEncoder` in Scikit-Learn), derived from the mathematical word *order*.

2. **One-Hot Encoding (`OneHotEncoder`):** Creates binary indicator columns ($0$ or $1$) for each unique category. Best when categories are unordered (nominal) and category cardinality is small (e.g. `ocean_proximity`).
3. **Embeddings:** When a categorical feature has hundreds or thousands of unique values (high cardinality), one-hot encoding produces massive, sparse tables. Embeddings project categories into dense, low-dimensional vectors where semantic similarities can be learned.

---

## 6. Custom Transformers & Scikit-Learn Design Principles

Scikit-Learn's API relies on consistent, unified design principles:
1. **Estimators:** Objects that estimate parameters from data via `fit(X, y=None)`. All ML algorithms and scalers/imputers are estimators. Learned parameters are stored with a trailing underscore (e.g. `imputer.statistics_`).
2. **Transformers:** Estimators capable of transforming a dataset via `transform(X)` or `fit_transform(X)`.
3. **Predictors:** Estimators capable of making predictions on new data via `predict(X)` (e.g. `LinearRegression`).

### Custom Transformer Class: `CombinedAttributesAdder`
- Wraps custom feature-engineering cleanup into a standardized Scikit-Learn class so it can participate seamlessly inside automated pipelines.

#### What the Class Needs:
- **`fit(self, X, y=None)`**: Learns parameters from data. For simple mathematical transforms (like ratios), nothing needs to be learned, so it returns `self`.
- **`transform(self, X)`**: Executes the actual data transformation and returns the modified array.
- **`TransformerMixin`**: Inheriting from this mixin automatically provides `fit_transform(X)` for free.
- **`BaseEstimator`**: Inheriting from `BaseEstimator` (and avoiding `*args` / `**kwargs` in `__init__`) automatically gives you `get_params()` and `set_params()`.

> 📝 **Side Note / Answering the Open Question:**
> *"When would I write a custom transformer instead of a plain function?"*
> 1. **When your transform has parameters to learn from training data:** If a transformation computes a statistic from the data (like mean, median, vocabulary, or frequency) that must be remembered and applied identically to test data, you need an object with `fit()` and `transform()`.
> 2. **When you want Scikit-Learn to tune feature-engineering hyperparameters:** By exposing parameters like `add_bedrooms_per_room=True/False` in `__init__`, tools like `GridSearchCV` can automatically test whether adding an engineered feature actually improves model accuracy!
> 3. **When a plain function is enough:** If a transformation is purely stateless (no parameters learned, no hyperparameters to tune), Scikit-Learn provides `FunctionTransformer` (e.g. `from sklearn.preprocessing import FunctionTransformer`), which lets you wrap a regular Python function directly into a pipeline without writing a class from scratch.

---

## 7. Feature Scaling

Machine learning algorithms typically perform poorly when numerical input attributes have vastly different scales (e.g. `total_rooms` ranges from 6 to 39,320, while `median_income` ranges from 0.5 to 15).

### Two Common Methods:
1. **Min-Max Scaling (Normalization):**
   - Rescales values into a fixed range (typically $[0, 1]$):
     $$x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
   - Available via Scikit-Learn's `MinMaxScaler`.
   - *Limitation:* Highly sensitive to extreme outliers.
2. **Standardization:**
   - Subtracts the mean and divides by standard deviation (unit variance):
     $$x_{\text{standardized}} = \frac{x - \mu}{\sigma}$$
   - Available via Scikit-Learn's `StandardScaler`.
   - Results in zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
   - *Advantage:* Much less affected by outliers than Min-Max scaling.

> 📝 **Golden Rule for Scaling:**
> Just like imputers, **never fit scalers on the full dataset**. Fit scalers on the **training set only**, then use the fitted scaler to transform validation, test, and production data.

---

## 8. Transformation Pipelines

Data preparation requires multiple transformation steps executed in the exact sequence.
1. **Pipeline**: Automates a sequential chain of data preparation and transformation steps so we don't apply them manually. All estimators except the last must be transformers.
2. **ColumnTransformer**: Applies distinct transformations to different subsets of columns in parallel and merges the output (e.g. numerical pipeline on numeric features vs. `OneHotEncoder` on categorical features).

- **Example**: A `num_pipeline` handles numerical columns by chaining `SimpleImputer`, `CombinedAttributesAdder`, and `StandardScaler`. A `ColumnTransformer` runs `num_pipeline` on numerical features and `OneHotEncoder` on `ocean_proximity` simultaneously to produce `housing_prepared`.

---

## 9. Select and Train a Model

After data preparation, we train and evaluate candidate baseline models:
- **Linear Regression**: A simple model assuming a linear relationship. Prone to **underfitting** if relationships are complex or non-linear.
- **Decision Tree Regressor**: A powerful non-linear model capable of fitting complex surfaces, but highly prone to **severe overfitting**.
- **K-Fold Cross-Validation**: Splits the training set into $K$ folds to measure generalization error without touching the test set.
- **Random Forest Regressor**: An ensemble model averaging predictions across multiple randomized decision trees.

> 📝 **Core Training Rules:**
> - **Don't trust training score alone**: A $0.00$ training error is a symptom of **overfitting**, not perfection.
> - **Try multiple baselines**: Always compare simple baselines (Linear Regression) against non-linear models and ensembles before narrowing down.

---

## 10. Fine-Tune the Model

Once a promising model is chosen (e.g. Random Forest), tune its hyperparameters for optimal performance:
1. **Grid Search (`GridSearchCV`)**: Evaluates all combinations of specified hyperparameters using cross-validation. Best for a small, targeted search space.
2. **Randomized Search (`RandomizedSearchCV`)**: Evaluates a fixed budget of random hyperparameter combinations. Best when searching over large continuous ranges.
3. **Ensemble Methods**: Combines predictions from multiple top-performing models (often outperforms individual models).
4. **Feature Importance**: Inspects which features contribute most to the predictions (e.g. `median_income` $\approx 36\%$, `INLAND` $\approx 15\%$) to drop uninformative features.
5. **Final Evaluation**: Evaluates the final tuned pipeline on the **test set once** to measure real-world generalization error (and computes a 95% confidence interval).

> ⚠️ **Golden Rule:** **Never tune hyperparameters on the test set!** The test set should only be evaluated once at the very end.

---

## 11. Launch, Monitor, and Maintain Your System

- **Deploy**: Package the full preprocessing pipeline and model into production (REST API, microservice, or web application).
- **Monitor Performance**: Proactively track live prediction quality, input data drift, and latency to detect performance degradation before users do.
- **Automate Retraining**: Build recurring pipelines to periodically retrain models on fresh data.
- **Model Snapshots & Rollback**: Store regular snapshots of trained weights and datasets to quickly roll back if a newly deployed model behaves unexpectedly.

---

## 12. Python & Scikit-Learn Syntax De-Mystified (Beginner's Rosetta Stone)

If Python syntax feels overwhelming or confusing, use this quick reference table to translate abstract code into plain English:

### A. Pandas Table Indexing & Manipulation
| Python Syntax | Plain English Translation |
| :--- | :--- |
| `housing["median_income"]` | Selects a single column as a 1D Pandas `Series`. |
| `housing[["col1", "col2"]]` | Double brackets! The outer `[...]` indexes into the table; the inner `["col1", "col2"]` is a list of column names. Returns a 2D `DataFrame`. |
| `housing.iloc[:5]` | **Integer location**: selects the first 5 rows (indices 0, 1, 2, 3, 4) by positional order, exactly like Python list slicing. |
| `housing.loc[train_indices]` | **Label location**: selects rows matching explicit index names/IDs (e.g. ID `12655`), regardless of where they sit in the table. |
| `df.drop("label", axis=1)` | Drops a **column** (`axis=1` = columns, `axis=0` = rows). Memory tip: `axis=1` stands tall like a vertical column `\|`. Returns a fresh copy without modifying the original. |
| `pd.cut(x, bins=[...], labels=[...])` | Chops continuous numeric values into discrete bucket categories (like grading test scores into A, B, C, D, F). |

### B. NumPy Array Slicing & Operations
| Python Syntax | Plain English Translation |
| :--- | :--- |
| `X[:, 3]` | 2D matrix slice: `[rows, columns]`. The colon `:` means "all rows"; `3` means column index 3 (the 4th column, since Python counts from 0). |
| `np.c_[array1, array2]` | **Column-stack**: glues 1D or 2D arrays horizontally side-by-side into adjacent columns. Memory tip: `c_` stands for **columns**. |
| `np.sqrt(x)` | Calculates the square root of every single number in an array. |
| `np.inf` | Positive infinity ($\infty$). Used in binning so any value above 6.0 falls into the highest category bucket. |

### C. Scikit-Learn API & Object-Oriented Patterns
| Python Syntax / Pattern | Plain English Translation |
| :--- | :--- |
| `estimator.fit(X, y)` | **Learn**: Analyzes the training data to calculate and store internal parameters (e.g., calculates column medians or regression slope weights). |
| `transformer.transform(X)` | **Apply**: Modifies data using the parameters previously learned during `fit()` (e.g., replaces missing values with learned medians). |
| `transformer.fit_transform(X)` | Efficiently runs `fit()` and then `transform()` on the training set in a single optimized pass. |
| `predictor.predict(X)` | Uses trained weights to predict target values on new input features. |
| Trailing underscore (e.g. `imputer.statistics_`, `grid_search.best_params_`) | **Scikit-Learn Rule**: Any attribute ending with a trailing underscore `_` was **learned from data during `.fit()`**. Hyperparameters passed by you do *not* have a trailing underscore. |
| `BaseEstimator` | Inheriting from this gives your custom class free `.get_params()` and `.set_params()` methods, enabling it to work seamlessly with `GridSearchCV`. |
| `TransformerMixin` | Inheriting from this automatically creates a free `.fit_transform()` method just by defining `fit()` and `transform()`. |
| `def __init__(self, ...):` | The constructor method. Runs automatically when you instantiate a class (e.g. `adder = CombinedAttributesAdder()`). `self` refers to that specific instance. |

### D. Python Idioms & Scikit-Learn Quirks
| Python Syntax | Plain English Translation |
| :--- | :--- |
| `[round(p, 1) for p in predictions]` | **List Comprehension**: Shorthand for a `for` loop. Means: "For each prediction `p` in `predictions`, round it to 1 decimal place, and collect results into a new list." |
| `zip(scores, names)` | Pairs two lists together element-by-element like teeth on a zipper: `(score[0], name[0]), (score[1], name[1]), ...`. |
| `scoring="neg_mean_squared_error"` | Scikit-Learn convention maximizes score metrics (higher is better). Since error is something we want to minimize, Scikit-Learn negates MSE into a negative number. That's why we write `np.sqrt(-scores)` to turn it positive again before taking the square root. |
| `random_state=42` | A pseudo-random seed number. Guarantees that random splits, tree building, and shuffling produce the exact same numbers every time you run the cell. |

