"""
Week 06 · Day 30 — Part D: AI-Augmented Task
Prompt: "Explain Logistic Regression with Python example using sklearn on SUV dataset."
Model: Claude (claude-sonnet-4-6)
Evaluation in W06_D30_PartD.md
"""

# ── AI Output (pasted and verified) ───────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('suv_data.csv')

# Encode Gender
df['Gender'] = (df['Gender'] == 'Male').astype(int)

# Features and target
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].astype(int).values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Probability scores for first 5 test samples
probs = model.predict_proba(X_test_scaled)[:5]
print("\nProbability scores (first 5):")
for i, (p0, p1) in enumerate(probs):
    print(f"  Sample {i}: P(not buy)={p0:.3f}  P(buy)={p1:.3f}  → Predicted: {int(p1 >= 0.5)}")


# ── My verification additions ──────────────────────────────────

print("\n--- Verification ---")

# 1. Confirm scaler was NOT re-fit on test data (correct practice)
#    If it were re-fit, test mean would also be ~0 — that would hide leakage
print("Train scaled mean:", X_train_scaled.mean(axis=0).round(4), " (should be ~0)")
print("Test  scaled mean:", X_test_scaled.mean(axis=0).round(4),  " (not guaranteed ~0)")

# 2. Confirm predictions are binary (0 or 1), not probabilities
assert set(np.unique(y_pred)).issubset({0, 1}), "Predictions should be binary"
print("All predictions are binary:", True)

# 3. Sanity check: accuracy should be between 0.8 and 1.0 for this dataset
acc = accuracy_score(y_test, y_pred)
assert 0.80 <= acc <= 1.0, f"Unexpected accuracy: {acc}"
print(f"Accuracy in expected range [0.80, 1.00]: {acc:.4f} ✓")
