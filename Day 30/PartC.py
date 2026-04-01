"""
Week 06 · Day 30 — Part C: Interview Ready (Coding)
Q2: Train-test split and scaling
Written answers for Q1 and Q3 in W06_D30_PartC.md
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Q2: Train-test split and feature scaling ──────────────────

df = pd.read_csv('suv_data.csv')
df['Gender'] = (df['Gender'] == 'Male').astype(int)

X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].astype(int).values

# Step 1 — Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42     # reproducibility
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# Step 2 — Scale
# CRITICAL: fit only on X_train, then transform both
# Fitting on full data would leak test set statistics into training (data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)    # use SAME scaler, do NOT re-fit

print(f"\nBefore scaling — Age range: {X_train[:,0].min():.0f} to {X_train[:,0].max():.0f}")
print(f"After  scaling — Age range: {X_train_scaled[:,0].min():.2f} to {X_train_scaled[:,0].max():.2f}")
print(f"\nMean after scaling (should be ~0): {X_train_scaled.mean(axis=0).round(4)}")
print(f"Std  after scaling (should be ~1): {X_train_scaled.std(axis=0).round(4)}")
