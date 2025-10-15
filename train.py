import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

df = pd.read_parquet("data/train.parquet")
X, y = df["text"], df["label"]

pipe = Pipeline([
  ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
pipe.fit(X, y)
dump(pipe, "models/baseline_lr.joblib")
