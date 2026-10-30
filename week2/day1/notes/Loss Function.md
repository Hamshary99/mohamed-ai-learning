# Loss Functions & Optimizers

## 1. Loss Function vs. Optimizer
* **Loss Function (The Compass):** Measures how far off predictions are from true labels. It **must be differentiable (smooth)** so Gradient Descent has a slope to follow (flat-step metrics like `Accuracy %` have a slope of $0$, leaving the model blind).
* **Optimizer (The Driver):** Uses the loss gradients from backpropagation to update weights in the downhill direction.

---

## 2. Optimizers Explained Simply

### 1. Stochastic Gradient Descent (SGD)
* **Idea:** Picks 1 random example (or a small mini-batch) to check the current slope and take a step right away—**no memory of past steps**, and the **same learning rate** for every weight.
* **Trade-Off:** Fast and handles huge datasets easily, but in narrow valleys with steep walls and a flat floor, it bounces violently wall-to-wall (zig-zagging) and gets stuck on flat plateaus.

### 2. SGD + Momentum
* **Idea (Heavy Bowling Ball):** Remembers past velocity (momentum vector $m$) and adds the current gradient to it. Side-to-side wall bounces cancel out, while forward pushes add up—rolling across flat plateaus up to $\frac{1}{1-\beta} = \mathbf{10\times}$ **faster** (when $\beta = 0.9$).
* **Trade-Off:** Powers past flat plateaus and local dips much faster than SGD, but the heavy ball can **overshoot** the bottom and swing back and forth before settling.

### 3. AdaGrad (Adaptive Gradient)
* **Idea (Custom Step Size per Weight):** Tracks the running sum of **all past squared gradients** for each weight and shrinks the learning rate more on steep slopes than on flat slopes.
* **Trade-Off:** Great for simple quadratic problems and sparse data, but in deep neural networks the accumulated sum keeps growing forever—shrinking the learning rate down to $0$ and **freezing training too early**.

### 4. RMSProp & AdaDelta
* **Idea (AdaGrad with Short-Term Memory):**
  * **RMSProp:** Fixes AdaGrad's freezing problem by using an exponential decay rate ($\beta \approx 0.9$) to only track **recent** squared gradients (forgetting old slopes).
  * **AdaDelta:** A sister method that also restricts past gradients to a recent window and removes the need to manually pick a learning rate.
* **Trade-Off:** Fixes premature stopping in deep networks, but only adjusts step *sizes*—it lacks Momentum's heavy-ball directional speed.

### 5. Adam (Adaptive Moment Estimation) & Nadam
* **Idea (Momentum + RMSProp Combined):**
  1. **1st Moment (Momentum):** Decaying average of past **gradients** ($\beta_1 = 0.9$) $\rightarrow$ builds directional speed.
  2. **2nd Moment (RMSProp):** Decaying average of past **squared gradients** ($\beta_2 = 0.999$) $\rightarrow$ adapts step size per weight.
  3. **Bias Correction:** Boosts both trackers on the first few steps so they aren't stuck near $0$ at the start.
  * *(Nadam = Adam + Nesterov look-ahead momentum to brake sooner before overshooting).*
* **Trade-Off:** The **default choice** for deep learning—fast and works out-of-the-box ($\eta = 0.001$), though it uses more memory and can sometimes generalize slightly worse than tuned `SGD + Momentum` on CNNs.

---

## 3. Comparison Table

| Optimizer | Simple Intuition | What It Tracks | Trade-Off (Pros vs. Cons) | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | Blindfolded single step using only current slope. | Current gradient only. | **+** Fast, low memory.<br>**−** Zig-zags in valleys; crawls on plateaus. | Simple baselines, massive datasets with tight memory. |
| **SGD + Momentum** | Heavy bowling ball building speed downhill ($10\times$ faster). | Past **gradients** (velocity). | **+** Rolls over plateaus; stops zig-zagging.<br>**−** Can overshoot the minimum. | **Computer Vision (CNNs / ResNets)** for strong generalization. |
| **AdaGrad** | Shrink step size on steep slopes, keep larger on flat ones. | **All** past **squared gradients**. | **+** Great for rare/sparse features.<br>**−** Learning rate shrinks to $0$ and freezes. | **Sparse data** (simple NLP / recommender systems); **not** deep nets. |
| **RMSProp / AdaDelta** | AdaGrad with short-term memory (only recent slopes). | **Recent** squared gradients. | **+** Fixes AdaGrad's freezing issue.<br>**−** No directional momentum buildup. | **RNNs**, reinforcement learning, non-stationary tasks. |
| **Adam / Nadam** | **Momentum + RMSProp** (heavy ball + smart step size). | Both past **gradients** & **squared gradients**. | **+** Fast, minimal tuning needed.<br>**−** Extra memory; can slightly overfit on CNNs. | **Default for Deep Learning** (MLPs, Transformers, LLMs). |
