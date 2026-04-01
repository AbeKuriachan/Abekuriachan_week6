# Week 06 · Day 30 — Logistic Regression on SUV Purchase Dataset

**Topics:** Logistic Regression · End-to-End ML Pipeline  
**Dataset:** [SUV Purchase Dataset (Kaggle)](https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset)

---

## What's in this repo

| File | Description |
|------|-------------|
| `PartAB.py` | Full ML pipeline — data loading, preprocessing, model training, evaluation, decision boundary & confusion matrix plots |
| `PartC.py` | Train-test split and StandardScaler implementation (Q2 coding answer) |
| `PartC.md` | Written answers — what is Logistic Regression, confusion matrix explanation |
| `PartD.py` | AI output (Claude) verified with assertion checks |
| `PartD.md` | Critical evaluation of AI output — what it got right and what it missed |
| `suv_data.csv` | Dataset (400 rows) |
| `confusion_matrix.png` | Heatmap of model predictions vs actual |
| `decision_boundary.png` | 2D decision boundary plot (Age vs Salary) |

---

## Results

| Split | Accuracy |
|-------|----------|
| 80/20 | 86.25% |
| 75/25 | 86.00% |
| 70/30 | 85.00% |

Model: `LogisticRegression` from sklearn · Features: `Age`, `EstimatedSalary` · Scaled with `StandardScaler`
