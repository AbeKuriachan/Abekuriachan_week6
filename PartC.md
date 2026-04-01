# Week 06 · Day 30 — Part C: Interview Ready (Written)

Code for Q2 is in `W06_D30_PartC.py`.

---

## Q1 — What is Logistic Regression? Is it Classification or Regression?

Logistic Regression is a **supervised binary classification** algorithm despite
having "regression" in its name. It models the **probability** that an input
belongs to a class using the **sigmoid function**:

```
P(y=1) = 1 / (1 + e^(−z))    where z = w₁x₁ + w₂x₂ + b
```

The sigmoid squashes any real number into (0, 1), which is interpreted as a
probability. A threshold (usually 0.5) converts this to a class label: if
P(y=1) ≥ 0.5 → predict 1, else predict 0.

**It is Classification, not Regression** because:
- Output is a discrete class label (0 or 1), not a continuous value
- It minimises **log loss** (cross-entropy), not MSE
- The decision boundary is a hyperplane that **separates classes**

In the SUV dataset: inputs are Age and EstimatedSalary; output is whether the
customer purchased (1) or not (0).

---

## Q3 — What is a Confusion Matrix?

A confusion matrix is a 2×2 table that breaks down model predictions into four
categories:

```
                    Predicted: 0       Predicted: 1
Actual: 0      True Negative (TN)   False Positive (FP)
Actual: 1      False Negative (FN)  True Positive (TP)
```

**For the SUV model (80/20 split):**
```
[[50  2]
 [ 9 19]]
TN=50  FP=2  FN=9  TP=19
```

**What each cell means:**
- **TN = 50** — correctly predicted "will not buy" → no action needed, correct
- **FP = 2** — predicted "will buy" but didn't → wasted marketing spend
- **FN = 9** — predicted "won't buy" but actually bought → missed opportunity
- **TP = 19** — correctly predicted "will buy" → successful targeting

**Derived metrics:**
- Accuracy = (TP + TN) / total = 69/80 = **86.25%**
- Precision = TP / (TP + FP) = 19/21 = **90.5%** (when we say "buy", we're right 90% of the time)
- Recall = TP / (TP + FN) = 19/28 = **67.9%** (we catch 68% of actual buyers)

The confusion matrix is more informative than accuracy alone — especially for
imbalanced classes, where accuracy can be misleadingly high.
