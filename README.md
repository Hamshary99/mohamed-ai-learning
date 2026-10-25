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

## 📋 Week 1 Dataset Execution Checklist (What to Apply Per Data)
*Standard Operating Procedure (SOP) aligned with Hands-On Machine Learning (Ch 1–9) & StudentGator Checkpoint #1 with Mohammed Orabi.*

Use this checklist whenever you explore a dataset in your sandbox ([`titanic_data_lab.ipynb`](titanic_data_lab.ipynb), California Housing, Stroke/Heart disease, or Kaggle competitions):

### 1. Problem Framing & Metric Strategy (Ch 1, 3)
- [X] **Define the task**: Supervised Binary Classification (`Survived` $\in \{0, 1\}$). Batch learning (static 891 rows, fits in memory).
- [X] **Establish the "Dumb Baseline"**: `DummyClassifier(strategy="most_frequent")` $\to$ **61.62% accuracy**. *(Benchmark floor: any real model must beat this without relying on blind majority guessing!)*.
- [X] **Choose the Metric Strategy**:
  - *Accuracy Trap*: Never rely on raw accuracy alone (class imbalance is ~62% died / 38% survived).
  - *Metric Choice*: Prioritize **$F_1$-Score** (balances precision & recall) and **ROC-AUC** to measure ranking quality across thresholds.

### 2. Exploratory Data Analysis (EDA) & Data Audit (Ch 2)
- [X] **Hygiene check**: Dropped uninformative identifier columns (`PassengerId`, `Ticket`) and sparse column `Cabin` (>77% missing). Kept raw `test_df['PassengerId']` safe for final Kaggle submission.
- [X] **Missingness audit**:
  - `Age` (~20% missing) $\to$ Impute with **Median** (robust against extreme age outliers).
  - `Embarked` (2 missing) $\to$ Impute with **Mode (`'S'`)** (Southampton represents ~72% of all boardings).
  - `Fare` (1 missing in test) $\to$ Impute with **Median**.
- [X] **Distributions & Skew**: Class distribution is **61.6% perished (`0`)** vs. **38.4% survived (`1`)**.
- [X] **Correlations & Key Insights**:
  - `Fare` ($+0.257$) and `Pclass` ($-0.338$) strongly drive survival.
  - *Why `Pclass` is negative*: Coded $1=\text{1st Class (Rich)}$, $2=\text{Mid}$, $3=\text{Poorest}$. Higher class number $\to$ lower survival.
  - *Survival rates by class*: **1st Class: 63.0%** | **2nd Class: 47.3%** | **3rd Class: 24.2%** (Rich were $>2.5\times$ more likely to survive).

### 3. Splitting Strategy (Prevent Data Snooping Bias) (Ch 2)
- [X] **Why split if `test.csv` exists?**: Kaggle's `test.csv` has **no labels (`Survived`)**—it is only for competition submission. To test accuracy, $F_1$, and check for overfitting locally, we must split `train_df` into local training and validation sets!
- [X] **Stratified Split**: Split 80/20 with `stratify=y` (`random_state=42`) to guarantee identical ~38.4% survival proportions in both `X_train` and `X_val`.
- [X] **Cross-Validation**: Use 5-fold cross-validation on `X_train` during hyperparameter tuning.

### 4. Preprocessing & Feature Engineering (Complete ✅)
- [X] **Numerical Imputation**: Imputed missing `Age` and `Fare` using training set medians (outlier robust).
- [X] **Categorical Imputation**: Imputed missing `Embarked` with the statistical mode (`'S'`).
- [X] **Feature Engineering (Domain)**:
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone = (FamilySize == 1)` (revealed that solo travelers survived at ~30% vs ~50% for families).
- [X] **Feature Engineering (Extraction)**:
  - Extracted `Title` from `Name` and grouped into 5 categories: `Mr`, `Miss`, `Mrs`, `Master`, `Rare`.
  - Discovered that young boys (`Master`) survived at **57.5%** compared to adult men (`Mr`) at **15.7%**!
- [X] **Encoding**:
  - Binary mapped `Sex` (`male=0`, `female=1`).
  - One-hot encoded nominal columns `Embarked` and `Title` (`drop_first=True`).
- [X] **Feature Scaling (Next step for models)**: Applied `StandardScaler()` via `make_pipeline()` to distance-sensitive models (Logistic Regression). Tree-based models (RF, XGBoost) are scale-invariant.

*(Note: Packaging into a Scikit-Learn `ColumnTransformer` is handled in Phase 8 for the Day 5 SOLID refactoring).*

### 5. Model Arena ("Horse Race" across Week 1 Algorithms) (Ch 4–7)
- [X] Fit and compare 3+ algorithms required for Week 1 deliverable:
  1. `DummyClassifier(strategy="most_frequent")` (Baseline floor: 61.62%)
  2. `LogisticRegression(max_iter=1000)` (Linear baseline)
  3. `RandomForestClassifier(n_estimators=150, max_depth=5)` (Bagging ensemble)
  4. `XGBClassifier(n_estimators=100, learning_rate=0.08)` (Gradient boosting)
- [X] Compare in a unified leaderboard: `Train Acc`, `Val Acc`, `Overfit Gap`, `F1 Score`, `ROC-AUC`.

### 6. Diagnostics: Bias vs. Variance & Overfitting (Ch 4)
- [X] **Compute Overfit Gap**: $\Delta = \text{Train Score} - \text{Val Score}$.
- [X] **Diagnose High Bias (Underfitting)**: Both train & test scores are low $\to$ Increase model capacity, decrease regularization ($C \uparrow$, `max_depth` $\uparrow$), add interaction/cluster features.
- [X] **Diagnose High Variance (Overfitting)**: High train score, lagging test score ($\Delta > 0.08$) $\to$ Diagnosed on default XGBoost ($\Delta = 0.102 > 0.08$); controlled by restricting `max_depth` (2–3) and adding regularization.
- [X] **Plot Validation Curve**: Track tree depth or $C$ vs. train/test scores to pinpoint the optimal generalization sweet spot.

### 7. Interpretability & Tuning (Ch 2, 3, 7)
- [ ] **Feature Importances**: Extract and plot `model.feature_importances_` from Random Forest or XGBoost.
- [ ] **Confusion Matrix**: Inspect `confusion_matrix(y_val, y_pred)` to find which specific cases the model confuses (Type I false alarms vs. Type II missed detections).
- [ ] **Hyperparameter Tuning**: Fine-tune the winning model using `GridSearchCV(cv=5)`.

### 8. Day 5: Clean Code & SOLID Refactoring
- [ ] **Decouple into 3 single-responsibility Python modules**:
  - `preprocessing.py` (SRP: Data ingestion, cleaning, feature engineering).
  - `train.py` (SRP: Model training loop, cross-validation, saving artifact with `joblib.dump`).
  - `evaluate.py` (SRP: Metric calculation, leaderboard table, confusion matrix).
- [ ] **Open/Closed Principle (OCP)**: Use a model dictionary registry so new algorithms can be plugged in without changing training/evaluation logic.

### 9. Checkpoint #1 Defense Guide (Meeting with Mohammed Orabi)
- [ ] **Walk through the pipeline**: Explain why each imputer, scaler, and encoder was selected.
- [ ] **Explain model choice**: Justify why $F_1$ / ROC-AUC was preferred over raw accuracy.
- [ ] **Explain bias vs. variance**: Demonstrate how you spotted overfitting and what hyperparameters you tuned to control it.
- [ ] **Code review**: Show refactored SOLID Python modules.

---

## 🎯 Week 1 Deliverable & Checkpoint #1 Requirements
*(What Mohammed Orabi will review at the end of Week 1)*

While **`titanic_data_lab.ipynb`** is your **interactive test field** (where you experiment, test hypotheses, and break code), your official Week 1 deliverable consists of two tangible components:

### Component 1: End-to-End Evaluation Notebook (`week1_deliverable.ipynb` or cleaned `titanic_data_lab.ipynb`)
Your finished notebook must clearly demonstrate:
1. **EDA & Data Insights**: Documented class imbalance (~62% died / 38% survived) and socio-economic findings (1st class $63\%$ vs 3rd class $24\%$).
2. **Feature Engineering**: `FamilySize`, `IsAlone`, and `Title` (capturing the $57.5\%$ survival of `Master` vs $15.7\%$ of `Mr`).
3. **Model Arena (3+ Algorithms)**: Direct comparison between:
   * **Logistic Regression** (Linear baseline)
   * **Random Forest** (Bagging ensemble)
   * **XGBoost** (Gradient boosting)
4. **Metrics Comparison Table**:
   * Tracking `Train Acc`, `Val Acc`, `Overfit Gap`, `Precision`, `Recall`, `F1 Score`, and `ROC-AUC`.
5. **Short Conclusion (4–6 sentences)**:
   * Which model won and by how much?
   * Which metric did you prioritize and why ($F_1$ / ROC-AUC due to class imbalance)?
   * Which features had the highest importance?
   * What would you improve next?

### Component 2: SOLID Modular Refactoring (`week1/day5/src/`)
Decouple the exploratory steps from your notebook into 3 clean, production-grade Python modules following the **Single Responsibility Principle (SRP)**:
```text
week1/day5/
├── src/
│   ├── preprocessing.py   # SRP: Data loading, median imputation, title extraction, one-hot encoding
│   ├── train.py           # SRP: Model factory (LogReg, RF, XGBoost), fitting, saving artifact (model.joblib)
│   └── evaluate.py        # SRP: Scoring functions, confusion matrix, metric report generation
└── main.py                # Orchestrates: load -> train -> evaluate
```

### Checkpoint #1 Checklist (Your 1:1 Meeting with Mohammed Orabi)
- [ ] Walk through the ML pipeline from raw data to predictions and explain all engineering choices.
- [ ] Explain why raw Accuracy is misleading on this dataset and justify choosing $F_1$ / ROC-AUC.
- [ ] Explain **Bias vs. Variance**: Show your train-vs-val scores and explain how you prevented overfitting (e.g., `max_depth`, regularization $C$, shrinkage).
- [ ] Walk through your refactored Python modules to demonstrate Clean Code and SOLID principles.
- [ ] Provide your GitHub repo link (`mohamed-ai-learning`), daily notes, and list of open questions.

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
