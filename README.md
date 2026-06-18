# KAN for Higgs Boson Classification
### University of Paris-Saclay — Internship Research

---

## Overview

This repository contains a complete, original implementation of **Kolmogorov-Arnold Networks (KAN)** for Higgs boson signal/background classification on the FAIR Universe HiggsML dataset.

Original contributions include model implementation, training methodology, systematic uncertainty robustness, and physics interpretability analysis.

---

## Research Questions

1. Can KAN match or exceed XGBoost/MLP on tabular HEP data?
2. Does adaptive grid placement improve KAN performance?
3. Is KAN more robust under systematic energy scale uncertainties?
4. Can KAN's learned activation functions reveal physics structure?

---

## Key Contributions

| Contribution | Description |
|---|---|
| **PyTorch KAN from scratch** | Full B-spline implementation with Cox-de Boor recursion |
| **Adaptive grid update** | Redistributes knot positions to data quantiles during training |
| **Grid extension (coarse→fine)** | Progressive refinement from G=3 to G=10 |
| **L1 spline regularisation** | Encourages sparse, interpretable edge functions |
| **Mixed precision training** | AMP on CUDA for ~2x speedup |
| **Bootstrap confidence intervals** | AUC and AMS reported as mean ± σ |
| **Systematic robustness** | Evaluates all models under energy scale shifts γ ∈ [0.8, 1.2] |
| **Adversarial decorrelation** | Pivoting method (Louppe et al., 2017) |
| **KAN interpretability** | Spline activation plots + feature importance |

---

## Dataset

**FAIR Universe — blackSwan_data** (systematic variation scenario)
- 2,000,000 events, 28 features
- Signal: H→ττ | Background: Z→ττ, ttbar, diboson
- Source: [FAIR Universe HiggsML Challenge](https://fair-universe.lbl.gov)

To download the dataset:
```bash
python download_data.py
```

---

## Environment

```
Python   3.12
PyTorch  2.11  (CUDA 12.8)
XGBoost  3.2
scikit-learn, pandas, pyarrow, seaborn, tqdm
```

Install all dependencies:
```bash
pip install torch xgboost scikit-learn pandas pyarrow seaborn tqdm
```

---

## How to Run

```bash
# 1. Download dataset
python download_data.py

# 2. Open notebook
jupyter lab KAN_Higgs_Internship_output_1132057.ipynb

# 3. All cells are pre-executed — results are already visible
```

To re-run from scratch using `KAN_Higgs_Internship.ipynb`, set in the config cell:
```python
MAX_EVENTS  = 2_000_000
EPOCHS_MAIN = 500
```

---

## Results

All metrics are mean ± σ over 200 bootstrap resamples on the held-out test set (300,000 events).
AMS uses LHC-normalised weights scaled to the full 10 fb⁻¹ luminosity (s = 1,015 expected signal events, b = 1,050,370 background events — the Zenodo "LHC Events" counts, #15131565).

### Full experiment: 2M events · 500 epochs · KAN hidden=[128], G=8, k=2

| Model | Params | AUC ± σ | AMS ± σ | Max AUC drop |
|---|---|---|---|---|
| XGBoost | — | 0.8533 ± 0.0008 | 2.2643 ± 0.0173 | 0.0521 |
| KAN (base, G=8, k=2) | 39,681 | 0.8472 ± 0.0008 | 2.0874 ± 0.0165 | 0.0450 |
| KAN + grid extension | 50,433 | 0.8432 ± 0.0008 | 2.1100 ± 0.0167 | **0.0268** |
| KAN-Adversarial | 39,681 | 0.8436 ± 0.0008 | 2.1456 ± 0.0178 | 0.0445 |

**Max AUC drop**: AUC degradation under momentum energy scale shifts γ ∈ [0.8, 1.2] — lower is more robust.

---

## References

- Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 (2024)
- Liu et al., *KAN 2.0*, arXiv:2408.10205 (2024)
- Louppe et al., *Learning to Pivot with Adversarial Networks*, NeurIPS 2017
- Cowan et al., *Asymptotic formulae for likelihood-based tests*, arXiv:1007.1727
- FAIR Universe HiggsML Dataset, Zenodo:10.5281/zenodo.15131565
