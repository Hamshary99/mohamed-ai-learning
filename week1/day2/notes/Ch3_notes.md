## The MNIST Dataset & Setup
MNIST is a benchmark dataset consisting of 70,000 small grayscale images (28×28 pixels) of handwritten digits (0 through 9).

Letting you know that the data is pre-shuffled so that all the cross validation folds are kinda similar distribution of digits. Algorithms sensitive to the order of the digits might perform poorly if not shuffled.

When **NOT** to shuffle is for time based data like history or stocks.

* **Train/Test Split**: 60,000 training instances, 10,000 test instances.
* **Shuffling**: Always shuffle training data unless working with time-series data to ensure balanced CV folds and prevent sequence bias.

---

## Training a Binary Classifier

Binary classifier is predicting two sets of classes. (True/False).
**SGDClassifier**: Handles classification instances one by one across the data set and it works great with large sets of data. Though it relies on the shuffle of 42 rounds (np.random)