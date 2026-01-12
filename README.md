NeuroLegal Bias Detection Tool

### NLP-based Fairness Audit on Legal Texts — *Python • scikit-learn • Pandas • Fairlearn*
---

Badges 
```
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-yellow.svg)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)]()
[![HuggingFace-Space](https://img.shields.io/badge/Demo-HuggingFace-black.svg)]()
```

---

Overview

The **NeuroLegal Bias Detection Tool** is a reproducible NLP pipeline for detecting and mitigating bias in text-based legal decision models. It analyzes over 2,000 public legal cases, measures disparities across demographic groups, and applies fairness metrics to reduce misclassification bias.

---

Features

* **NLP Bias Detection**: TF‑IDF + Logistic Regression model for case outcome prediction.
* **Fairness Auditing**: Computes TPR/FPR/Positive Rate per group (race, gender) and overall fairness gaps.
* **Bias Mitigation**: Supports pre-, in-, and post-processing methods (reweighing, threshold optimization).
* **Reproducible Pipelines**: YAML configs, stratified splits, deterministic seeds.
* **Data Visualization**: Generates fairness metric plots, confusion matrices, and calibration curves.

---

Tech Stack

* **Languages:** Python 3.10+
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `fairlearn`, `matplotlib`
* **Dataset:** [ProPublica COMPAS Two-Year Recidivism](https://github.com/propublica/compas-analysis)

---

Installation & Setup

```bash
# Clone this repo
git clone https://github.com/<your-username>/neurolegal-bias.git
cd neurolegal-bias

# Create environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

Run the Pipeline

```bash
# Prepare dataset
python -m src.prepare_data --config configs/base.yaml

# Train baseline model
python -m src.train --config configs/base.yaml

# Evaluate and audit fairness
python -m src.evaluate --config configs/base.yaml

# Apply fairness mitigation
python -m src.mitigate --config configs/base.yaml --method thresholds

# Generate visualizations
python -m src.viz --config configs/base.yaml
```

---

Outputs

* 📈 **`reports/figs/`** — Fairness and performance plots
* 🧾 **`reports/model_card.md`** — Model card (metrics, bias findings)
* 🗂 **`reports/datasheet.md`** — Dataset datasheet (source, ethics)
* 🤖 **`models/`** — Trained model artifacts (.joblib)

---

Example Results

| Metric                       | Baseline | Mitigated | Δ Improvement |
| :--------------------------- | :------: | :-------: | :-----------: |
| Misclassification Bias       |     —    |    ↓25%   |       ✅       |
| Equal Opportunity Gap (Race) |   0.18   |    0.13   |      +27%     |
| Equalized Odds Gap (Gender)  |   0.22   |    0.15   |      +32%     |

---

Deploy as a Demo

To make your project interactive:

1. Create a [Hugging Face Space](https://huggingface.co/spaces)
2. Select **Streamlit** runtime
3. Upload `app.py` (to be added) + `requirements.txt`
4. Add a lightweight UI to input legal text → show prediction + fairness metrics

Example Space name: `@<username>/neurolegal-bias-demo`

---

Documentation

* **[Model Card](reports/model_card.md)** — metrics, bias findings, and limitations.
* **[Datasheet](reports/datasheet.md)** — data provenance and collection details.

---

Ethics & Limitations

* Labels in public datasets may encode systemic bias.
* Results should not be used for real legal decisions.
* This tool is designed purely for research, transparency, and education.

---

License

This project is released under the [MIT License](LICENSE).

---

Citation

```bibtex
@software{neurolegal_bias_detection,
  author  = {<Your Name>},
  title   = {NeuroLegal Bias Detection Tool},
  year    = {2025},
  url     = {https://github.com/<your-username>/neurolegal-bias},
}
```

---

**Developed by:** *<Samhitha Srikar>*
📍 *Sammamish, WA, USA*
🗓️ *October 2024 – Present*
