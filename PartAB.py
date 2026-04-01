"""
Week 06 · Day 30 — Parts A & B
Logistic Regression End-to-End Pipeline on SUV Purchase Dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# PART A — DATA LOADING, PREPROCESSING, TRAINING

# ── A1. Load & Explore ────────────────────────────────────────
df = pd.read_csv('suv_data.csv')

print("=== Dataset Overview ===")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df['Purchased'].value_counts())


# ── A2. Preprocessing ─────────────────────────────────────────

df['Gender'] = (df['Gender'] == 'Male').astype(int)

X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].astype(int).values

print("\nFeatures shape:", X.shape)
print("Target shape:  ", y.shape)


# ── A3. Train-Test Split (80/20) ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")


# ── A4. Feature Scaling ───────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on train only
X_test_s  = scaler.transform(X_test)        # apply same scale to test

print("Scaled train mean (should be ~0):", X_train_s.mean(axis=0).round(4))
print("Scaled train std  (should be ~1):", X_train_s.std(axis=0).round(4))


# ── A5. Model Training ────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train_s, y_train)

print("\nModel coefficients:", model.coef_)
print("Model intercept:   ", model.intercept_)


# PART B — EVALUATION, VISUALIZATION, IMPROVEMENT

# ── B1. Model Evaluation ──────────────────────────────────────
y_pred = model.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print("\n=== Model Evaluation (80/20) ===")
print(f"Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased']))

# Confusion matrix interpretation
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives  (correctly predicted 'not buy'): {tn}")
print(f"False Positives (predicted 'buy', actually not):  {fp}")
print(f"False Negatives (predicted 'not buy', actually bought): {fn}")
print(f"True Positives  (correctly predicted 'buy'):      {tp}")


# ── B2. Visualization — Decision Boundary ────────────────────
def plot_decision_boundary(X_scaled, y, model, scaler, title, filename):
    h = 0.01
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                          c=y, cmap='RdYlBu', edgecolors='k', s=30)
    ax.set_xlabel('Age (scaled)')
    ax.set_ylabel('Estimated Salary (scaled)')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Purchased')
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved: {filename}")


X_all_s = scaler.transform(X)
plot_decision_boundary(X_all_s, y, model, scaler,
                        'Logistic Regression — Decision Boundary (SUV Purchase)',
                        'decision_boundary.png')

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Not Purchased', 'Purchased'])
ax.set_yticklabels(['Not Purchased', 'Purchased'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                fontsize=16, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.close()
print("Saved: confusion_matrix.png")


# ── B3. Compare Different Split Sizes ─────────────────────────
print("\n=== Split Size Comparison ===")
print(f"{'Split':<12} {'Train':>6} {'Test':>6} {'Accuracy':>10}")
print("-" * 38)
for ts in [0.20, 0.25, 0.30]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42)
    sc2 = StandardScaler()
    m   = LogisticRegression(random_state=42).fit(sc2.fit_transform(Xtr), ytr)
    a   = accuracy_score(yte, m.predict(sc2.transform(Xte)))
    split = f"{int((1-ts)*100)}/{int(ts*100)}"
    print(f"{split:<12} {len(Xtr):>6} {len(Xte):>6} {a:>10.4f}")

# 80/20 gives the highest accuracy (86.25%) — larger training set helps
# 70/30 drops slightly to 85.00% — less training data, slightly weaker boundary
