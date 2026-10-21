# Chapter 3 — Classification: MNIST Notes

## 1. The MNIST Dataset & Setup

- **The Dataset**: Benchmark dataset consisting of 70,000 small grayscale images (28×28 pixels = 784 features) of handwritten digits (0 through 9).
- **Train/Test Split**: 60,000 training instances, 10,000 test instances.
- **Shuffling**: Training data is pre-shuffled so all cross-validation folds have a similar distribution of digits (prevents sequence bias where algorithms sensitive to digit order perform poorly).
  - **When NOT to shuffle**: Time-based data (like stocks or history) where the chronological sequence order matters.

---

## 2. Training a Binary Classifier

- **Binary Classification**: Predicting between two classes (`True` / `False`, e.g., "5" vs. "not-5").
- **`SGDClassifier`**:
  - Handles classification instances one by one across the dataset, making it fast on large data and well-suited for **online learning**.
  - **Why `random_state=42`?**: SGD relies on randomness during training (it randomly picks samples). Setting `random_state=42` is a **seed number** to freeze the random generator for reproducible results (it does *not* mean 42 rounds of shuffling; `42` is just a fixed seed value).

---

## 3. Performance Measures & Why Accuracy Fails

Measuring classification performance is much trickier than regression (where we used RMSE / MAE):

- **The Illusion**: Evaluating `SGDClassifier` using 3-fold cross-validation gives **~95–96% accuracy**.
- **The Trap**: Only ~10% of the images are 5s (90% are not 5s). A dumb classifier (`DummyClassifier`) that *always* predicts `False` ("never-5") achieves **~91% accuracy** without learning anything!
- **Conclusion**: The SGD classifier's 95.7% accuracy is really only ~4.7% better than blindly guessing "not-5".
- We are always prone to **data snooping bias** when we evaluate the model multiple times. So never evalute on test set, use cross validation set predictions instead.

Though we can use our costume cross validation score if the scikit learn does not provide the metrics we need. 

> ⚠️ **Golden Rule:** **Accuracy is often misleading on skewed/imbalanced datasets.** Never rely on accuracy alone.
> Never rely on raw accuracy alone; look at confusion matrices, precision, recall, and ROC AUC.

---

## 4. Confusion Matrix

Confusion matrix during classification is by providing a row of actual classes against the predicted classes in the row.

`cross_val_predict()` generates predictions for each instance using a model trained on other folds.

**4 Quadrants**: 
* **TN**: Correctly identified negative instances.
* **FP** (Type I Error): False alarm (wrongly flagged as positive).
* **FN** (Type II Error): Missed detection (wrongly flagged as negative).
* **TP**: Correctly identified positive instances.

> **NOTES**: I still need help from Orabi to figure out on what bases do we generate the predictions and putting it in confusion matrix.

---

## 5. Precision, Recall, and $F_1$ Score

Instead of looking at the full matrix, we compress model performance into focused summary metrics:

- **Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  Overall fraction of correct predictions across all classes (misleading on imbalanced data).

- **Precision**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  **Quality of positive claims:** Out of all instances the model predicted as positive (e.g., "this is a 5"), what percentage was actually correct? *(Goal: minimize false alarms / FP)*.

- **Recall (Sensitivity)**:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  **Coverage of actual positives:** Out of all the real positive instances that exist (all actual 5s), what percentage did the model successfully catch? *(Goal: minimize missed detections / FN)*.

- **$F_1$ Score**:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{TP}{TP + \frac{FP + FN}{2}}$$
  **Harmonic mean** balancing precision and recall into a single metric. Unlike a simple arithmetic average, the harmonic mean gives much more weight to low values—so $F_1$ is only high if *both* precision and recall are high.

  > $F_1$ really favors that precision is as similar as to recalls, but some applications need a significant precision metrics or vice versa. 

**Favoring Recalls example**: suppose you train a classifier to detect shoplifters on surveillance images: it is probably fine if your classifier has only 30% precision as long as it has 99% recall (it will raise many false alarms, but it will not miss many shoplifters).

**Favoring Precision example**: suppose you train a classifier to detect good videos for kids : it is probably fine if your classifier has only 30% recall as long as it has 99% precision (it will miss many good videos but it will not classify too many bad  videos as good).

**The Trade-off**: You cannot maximize both simultaneously; increasing precision lowers recall and vice-versa.

---

## 6. The Precision/Recall Trade-off & Threshold Tuning


Algorithms like `SGDClassifier` imposes a computing score that if it exceeds a threshold the instance will be predicted as positive else it will be predicted as negative.

As we increase the threshold, the precision increases and the recall decreases, and vice versa.

$$Threshold \Uparrow \implies Precision \Uparrow, \quad Recall \Downarrow$$

$$Threshold \Downarrow \implies Precision \Downarrow, \quad Recall \Uparrow$$


>**The "99% Precision" Trap**: Anyone can create a 99% precision classifier by setting an extremely high threshold. Always ask: *"At what recall?"* A 99% precision model with 0.1% recall is useless.

> ⚠️ **Golden Rule:** **Accuracy is often misleading on skewed/imbalanced datasets.** Never rely on accuracy alone. Never rely on raw accuracy alone; look at confusion matrices, precision, recall, and ROC AUC.

---

## 7. The ROC Curve & ROC AUC

The receiver operating characteristic (ROC) curve plots the **True Positive Rate (Recall)** against the **False Positive Rate (FPR)** across all possible thresholds.

$$\text{TPR (Recall)} = \frac{TP}{TP + FN}$$

$$\text{FPR} = \frac{FP}{TN + FP} = 1 - \text{Specificity (True Negative Rate)}$$

- **The Trade-off**: Increasing Recall (TPR) inevitably increases False Positives (FPR).
- **The Ideal Goal**: The top-left corner (100% recall, 0% false alarms).
- **ROC AUC**: Area Under Curve from $0.5$ (pure coin flip) to $1.0$ (perfect). Our SGD gets $\approx \mathbf{0.9605}$.

---

> ⚠️ **Rule of Thumb (PR vs. ROC Curve):**
> Prefer the **PR curve** whenever the positive class is rare (imbalanced data) or when you care more about false alarms (FPs). Prefer the **ROC curve** otherwise.
>
> *Why does ROC look deceptively good on MNIST?* Because 90% of images are non-5s, making the denominator $TN + FP$ huge in the FPR equation. Even with hundreds of false alarms, FPR stays tiny. In contrast, the PR curve ($\frac{TP}{TP + FP}$) makes it clear that precision drops when catching more 5s.

---

### Comparing Models: `SGDClassifier` vs. `RandomForestClassifier`

- **`SGDClassifier`**: Uses `decision_function()` (raw scores) $\implies \text{ROC AUC} \approx 0.960$.
- **`RandomForestClassifier`**: Lacks `decision_function()`, so we use `predict_proba()` (probability of positive class: `[:, 1]`) $\implies \text{ROC AUC} \approx \mathbf{0.998}$.
- **Result**: Random Forest hugs the top-left corner much tighter, achieving **$\approx 99\%$ precision** and **$\approx 86.6\%$ recall**.

