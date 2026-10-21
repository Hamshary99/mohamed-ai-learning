# Chapter 1 — Machine Learning landscape

## Why Machine Learning?

**Machine Learning (ML)** helps when:

- Existing solutions need a lot of **hand-tuning**
- The problem is too complex for traditional hard-coded rules
- The system must **adapt to new data** (even large amounts)
- You want the machine to improve from examples, not from you writing every rule

ML systems can learn:

- **Online** or in **batches**
- With **supervised**, **semi-supervised**, or **unsupervised** data
- By comparing new inputs to old ones, or by finding the overall “style” of a group of inputs

---

## How much supervision?

ML systems can be classified by the **amount and type of supervision** they get during training.

### Supervised learning

**Supervised learning:** you train with **labeled data** — each example already has the correct answer (the **label**).

Common tasks:

- **Classification** — put examples into classes (spam / not spam)
- **Regression** — predict a number from features (price, life satisfaction, etc.)

### Unsupervised learning

**Unsupervised learning:** no labels. The model finds structure in the data on its own (you check the patterns afterward).

Common tasks:

- **Clustering** — group similar examples (e.g. hierarchical clustering)
- **Dimensionality reduction** — shrink many features into fewer ones without losing too much information
- **Anomaly detection** — spot odd cases compared to what was “normal” in training
- **Novelty detection** — expect mostly normal inputs; flag anything new/strange
- **Association rules** — find relationships across a large dataset (e.g. “people who buy X also buy Y”)

### Semi-supervised learning

**Semi-supervised learning:** most data is unlabeled; only some examples have labels.

Why it matters: full labeling is expensive. A common approach is to cluster unlabeled data first, then label a few groups.

### Reinforcement learning

**Reinforcement learning** is a different setup:

- The model is an **agent**
- It acts in an **environment**
- It follows a **policy** (the strategy / rules it uses)
- It gets a **reward** or **penalty** based on how good the action was

Loop:

1. Observe  
2. Follow policy  
3. Take action  
4. Get reward or penalty  
5. Update policy  
6. Repeat until the policy is good enough  

---

## Batch learning vs online learning

Another way to classify ML: can the system learn **incrementally** from a stream of data?

### Batch learning

**Batch learning (offline learning):**

1. Train on the full dataset  
2. Launch to production  
3. **Stop training**  

To adapt to new data: train a **new model from scratch** on old + new data, then replace the old model.

Can be automated, but retraining on bigger and bigger datasets costs hardware, time, and money — and may be impossible if the dataset is huge.

### Online learning

**Online learning:** the model updates as new data arrives, either one example at a time or in small groups called **mini-batches**.

Pros:

- Lighter on hardware  
- Can drop old data you no longer need, or keep some for **replay**

**Learning rate** = how sensitive the update is:

- **High** → adapts fast to new data, but may forget older patterns  
- **Low** → learns gradually, keeps older knowledge better  

Bad inputs can damage an online model, so it needs monitoring / supervision.

---

## Instance-based vs model-based learning

**Generalization** = doing well on **new, unseen** examples (not just the training set).

### Instance-based learning

**Instance-based learning:** remember the training examples, then compare a new case to them using a **similarity measure** (e.g. **k-Nearest Neighbors**).

Simple idea: “this looks like those past examples → same answer.”

### Model-based learning

**Model-based learning:** instead of memorizing every point, learn a general rule or formula from the data (merge datasets, find relations, then predict).

Example from the book: use GDP to predict life satisfaction with a line.  
**k-Nearest Neighbors** can also help: a country with similar GDP may have a similar happiness score.

Real results also depend on environment, employment, health, pollution, etc. — GDP alone is incomplete.

**Cost function** = how bad predictions are compared to the true values (distance from the trend line).  
**Training** = adjusting the model to make that cost as small as possible.

---

## Challenges of Machine Learning

Problems usually come from a **bad algorithm** or **bad data** (often both).

### Bad data

#### Insufficient quantity of data

Not enough examples for the model to learn a solid pattern.

#### Nonrepresentative training data

To **generalize**, training data must look like the real cases you’ll see later.  
If the sample is biased or too small, predictions will be off.

#### Poor-quality data

**Outliers** and **noise** hide the real pattern. A lot of ML work is cleaning: fix, drop, or decide what missing values mean.

#### Irrelevant features

Junk or unrelated columns hurt learning. Feature choice matters.

#### Overfitting the training data

**Overfitting:** the model is **too complex** for the data. It memorizes the training set (including noise) and fails on new data.

Looks great in training, terrible in production.

**Regularization** constrains the model to stay simpler — fights overfitting so it generalizes better.

#### Underfitting the training data

**Underfitting:** the opposite — the model is **too simple** to capture the real structure of the data.

---

## Big picture

- ML = make the machine better at a task using data  
- Systems can be supervised or not, online or batch, instance- or model-based  
- Typical project: gather a **training set** → feed a learning algorithm  
  - **Model-based:** tune **parameters** to fit the data, then predict  
  - **Instance-based:** store examples and generalize by similarity  
- Performance suffers if data is scarce, noisy, irrelevant, or nonrepresentative  
- Aim for a model that is neither too simple nor too complex  

---

## Testing and validating

The only reliable way to know if a model **generalizes** is to try it on **new cases**.

Split the data:

1. **Training set** — used to fit the model’s parameters  
2. **Validation set** (also called **holdout set** / **dev set**) — used to compare models and tune choices without touching the final test  
3. **Test set** — used **once at the end** to estimate real-world performance  

### Data mismatch

**Data mismatch:** training data is plentiful but doesn’t match production (e.g. clean web photos vs blurry phone photos).

**Train-dev set:** a small slice held out **from the training distribution** that you never train on. It helps tell **overfitting** apart from **data mismatch**.

#### How to diagnose performance drops

1. **Check the train-dev set**
   - **High error** → likely **overfitting** the training set  
   - Fix: simplify / regularize, get more data, clean noise  

2. **If train-dev looks good, check the validation set**
   - **High error on validation** → likely **data mismatch**  
   - Fix: make training data closer to production (transforms), or collect more representative data  

---

## Exercises (make sure you can answer these)

1. How would you define Machine Learning?  
2. Can you name four types of problems where it shines?  
3. What is a **labeled training set**?  
4. What are the two most common **supervised** tasks?  
5. Can you name four common **unsupervised** tasks?  
6. What type of ML would you use so a robot can walk in unknown terrains?  
7. What type of algorithm would you use to segment customers into groups?  
8. Is spam detection supervised or unsupervised?  
9. What is an **online learning** system?  
10. What is **out-of-core learning**?  
11. What type of learning relies on a **similarity measure**?  
12. Difference between a **model parameter** and a **hyperparameter**?  
13. What do model-based algorithms search for? What’s the usual strategy? How do they predict?  
14. Name four main challenges in Machine Learning.  
15. Great on training, poor on new data — what’s happening? Name three fixes.  
16. What is a **test set**, and why use it?  
17. Purpose of a **validation set**?  
18. What goes wrong if you tune hyperparameters on the **test set**?  
19. What is **repeated cross-validation**, and why prefer it to one validation set?  

What's hyperparameter?
What are cross-validation  and holdout validation?
What defines the 'irreducible error' portion of a model's total generalization error?