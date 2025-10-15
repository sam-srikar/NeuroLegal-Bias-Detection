# NeuroLegal-Bias-Detection
A machine learning project that identifies and analyzes bias in legal decision-making. This tool uses natural language processing (NLP) to examine court documents and detect linguistic or structural indicators of bias. Inspired by cognitive neuroscience principles.
# NeuroLegal Bias Detection Tool (COMPAS, scikit-learn, pandas)


**Goal**: Train an NLP baseline to predict 2-year recidivism using textual **charge descriptions** and audit/mitigate fairness across **race** and **sex**.


## Dataset
We use the ProPublica COMPAS two-year recidivism dataset (fields used include `c_charge_desc`, `sex`, `race`, `two_year_recid`).


> **Expected file:** `data/raw/compas-scores-two-years.csv`


## Quickstart
```bash
# 1) create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


# 2) configure
# configs/base.yaml already points to ./data and ./models by default


# 3) prepare data
python -m src.prepare_data --config configs/base.yaml


# 4) train baseline (TF-IDF + LogisticRegression)
python -m src.train --config configs/base.yaml


# 5) evaluate & audit fairness (overall + by race/sex)
python -m src.evaluate --config configs/base.yaml


# 6) optional mitigation (reweighing or per-group thresholds)
python -m src.mitigate --config configs/base.yaml --method thresholds


# 7) visualize before/after fairness metrics
python -m src.viz --config configs/base.yaml
