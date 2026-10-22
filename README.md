# mohamed-ai-learning
## StudentGator AI Engineering Internship: From ML to Production

A 4-week learning program covering Machine Learning Foundations, Deep Learning, Transformers, RAG, FastAPI, Docker, CI/CD, and GCP.

Organized by **week → day** (4 weeks × 5 days). Shared Python env stays at the root.

## Layout

```
ML/
  .venv/                      # one env for everything
  requirements.txt
  week1/
    day1/                     # current work (Ch1 + Ch2)
      notebooks/
      notes/
      data/
    day2/ … day5/             # empty — ready
  week2/ … week4/
    day1/ … day5/
```

## How to run notebooks

1. Open a notebook under `weekX/dayY/notebooks/`.
2. Kernel → **ML Sandbox (.venv)**.
3. Run with `Shift+Enter`.

From a notebook, day’s data is:

```python
pd.read_csv("../data/your_file.csv")
```

## What goes where

| When | Put it in |
|------|-----------|
| New day | `weekN/dayM/notebooks/` |
| Notes | `weekN/dayM/notes/` |
| Datasets | `weekN/dayM/data/` |

---

## ML Practice Playbook & Test Field Guide

When moving beyond textbook examples to test your knowledge on messy, real-world data:

### 1. Realistic Datasets to Practice On
* **Titanic** (`week1/day1/notebooks/Week1_Titanic_Starter.ipynb`): The gold standard for binary classification with missing data, mixed types, and feature engineering.
* **Stroke / Heart Disease Prediction** ([Kaggle / UCI](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)): Realistic tabular health records, imbalanced classes, great for testing Logistic Regression vs. SVM vs. Trees.
* **California Housing** (`week1/day1/data/`): Continuous tabular data for testing `LinearSVR`, `SVR(kernel="rbf")`, and Regression Trees.
* **OpenML Direct Load**: Pull datasets in 1 line without manual CSV downloads:
  ```python
  from sklearn.datasets import fetch_openml
  titanic = fetch_openml('titanic', version=1, as_frame=True)
  df = titanic.frame
  ```

### 2. Sources to Imitate & Learn From
* **Kaggle Notebooks**: Go to any dataset $\to$ **Code** tab $\to$ filter by **Most Upvoted**. Study how top practitioners clean data, structure `Pipelines`, and evaluate models.
* **Scikit-Learn Examples Gallery** ([scikit-learn.org/stable/auto_examples](https://scikit-learn.org/stable/auto_examples/index.html)): Clean, officially maintained scripts comparing models, plotting boundaries, and tuning hyperparameters.

### 3. The 4-Step Sandbox Workflow
1. **Load data**: Use `pandas` or `fetch_openml`.
2. **Build Preprocessing Pipeline**: `SimpleImputer` $\to$ `StandardScaler` / `OneHotEncoder`.
3. **Model Horse Race**: Fit 2–3 algorithms on the exact same preprocessed data (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`).
4. **Compare & Tune**: Evaluate with `cross_val_score(cv=5)` and tune hyperparameters ($C$, $\gamma$, `max_depth`).

---

## How to Learn Coding & ML (Without Memorizing Syntax)

> **Golden Rule**: Professional developers and Data Scientists **do not memorize syntax**. They remember **concepts, workflows, and mental models**, while referencing documentation for exact syntax.

### The 3-Step "Read $\to$ Break $\to$ Own" Learning Loop:
1. **Never copy-paste blindly, but never code from pure blank memory either**:
   - Keep a reference or cheatsheet open on half your screen.
   - Look at a block of code, understand **what each line does** in plain English.
2. **Type it out by hand**:
   - Typing builds motor memory. As you type, ask: *"What would happen if I removed this scaler? What if I change $C=100$ to $C=0.01$?"*
3. **Learn by Debugging Errors**:
   - When Python throws a `TypeError` or `ValueError`, that is **where actual learning happens**. Read the error message, trace where shapes mismatched, and fix it.
4. **Master the Scikit-Learn 3-Word Grammar**:
   - Scikit-learn has the most consistent API in programming:
     - `.fit(X, y)`: Learn patterns from data.
     - `.transform(X)`: Modify/scale features (transformers).
     - `.predict(X)`: Make predictions (classifiers/regressors).
   - Once you know this 3-word grammar, you already know 90% of the syntax for almost every algorithm in the library.
