# Week 06 · Day 30 — Part D: AI-Augmented Task (Evaluation)

Code is in `W06_D30_PartD.py`.

## Prompt
"Explain Logistic Regression with Python example using sklearn on SUV dataset."
Model: Claude (claude-sonnet-4-6)

## AI Output Summary

The AI produced a complete pipeline:
1. Load CSV → encode Gender → select features → split → scale → train → evaluate
2. Used `classification_report` for precision/recall/F1 breakdown
3. Added `predict_proba` to show probability scores — a useful addition beyond
   just class labels

## Critical Evaluation

**Is the code correct?**
Yes — all steps are implemented correctly. The scaler is fit only on `X_train`
and applied (not re-fit) on `X_test`, which is the correct practice to avoid
data leakage. The use of `random_state=42` throughout ensures reproducibility.

**Are the steps complete?**
Mostly. The AI covered: load, encode, split, scale, train, predict, evaluate
with confusion matrix and classification report. What was missing:
- No comparison across different split sizes (70/30, 75/25)
- No visualisation (decision boundary or confusion matrix heatmap)
- No explicit commentary on what the confusion matrix numbers mean in
  business terms (who are the false negatives — missed SUV buyers?)

**What I added:**
Three assertion-based verification checks:
1. Confirm the scaler was not re-fit on test data (mean of test set is not ~0)
2. Confirm predictions are binary integers, not probability floats
3. Confirm accuracy falls in a reasonable range for this dataset (0.80–1.00)

These turn the output from a demo into something closer to production-quality
code with built-in sanity checks.
