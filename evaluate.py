import pandas as pd, numpy as np
from joblib import load
from sklearn.metrics import f1_score, roc_auc_score

df = pd.read_parquet("data/test.parquet")
model = load("models/baseline_lr.joblib")
proba = model.predict_proba(df["text"])[:,1]
pred  = (proba >= 0.5).astype(int)

def metrics(y, yhat):
    return dict(
      f1=f1_score(y, yhat),
      auc=roc_auc_score(y, proba)
    )

overall = metrics(df["label"], pred)

by_group = {}
for g, sub in df.groupby("group"):
    y, yhat = sub["label"], (proba[sub.index] >= 0.5).astype(int)
    tp = ((y==1)&(yhat==1)).sum(); fn=((y==1)&(yhat==0)).sum()
    fp = ((y==0)&(yhat==1)).sum(); tn=((y==0)&(yhat==0)).sum()
    tpr = tp / (tp+fn+1e-9); fpr = fp / (fp+tn+1e-9); pos = yhat.mean()
    by_group[g] = {"TPR":tpr, "FPR":fpr, "PosRate":pos}

# compute gaps
groups = list(by_group)
tpr_gap = max(by_group[g]["TPR"] for g in groups) - min(by_group[g]["TPR"] for g in groups)
fpr_gap = max(by_group[g]["FPR"] for g in groups) - min(by_group[g]["FPR"] for g in groups)

print("overall:", overall)
print("tpr_gap:", tpr_gap, "fpr_gap:", fpr_gap)
