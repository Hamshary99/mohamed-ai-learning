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

> ⚠️ **Golden Rule:** **Accuracy is often misleading on skewed/imbalanced datasets.** Never rely on accuracy alone.


