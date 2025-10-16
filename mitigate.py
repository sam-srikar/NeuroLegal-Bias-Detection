import numpy as np, pandas as pd
from joblib import load

df = pd.read_parquet("data/val.parquet")
model = load("models/baseline_lr.joblib")
proba = model.predict_proba(df["text"])[:,1]

# simple grid search for per-group thresholds targeting equal TPR
thresholds = {}
target_tpr = 0.75  # or avg TPR baseline
for g, sub in df.groupby("group"):
    y = sub["label"].to_numpy()
    p = proba[sub.index].to_numpy()
    best = 0.5; best_diff = 1e9
    for t in np.linspace(0.2, 0.8, 61):
        yhat = (p>=t).astype(int)
        tpr = ((yhat==1)&(y==1)).sum() / (y.sum()+1e-9)
        diff = abs(tpr - target_tpr)
        if diff < best_diff:
            best_diff = diff; best = t
    thresholds[g] = best

# save thresholds, then apply them at test time
