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

The key KAN settings (in the config cell) are:
```python
MAX_EVENTS    = 2_000_000
KAN_HIDDEN    = [96, 96, 96]   # deep network — the tuned architecture that beats XGBoost
KAN_GRID_SIZE = 5
KAN_ORDER     = 3
LR_KAN        = 5e-4
```

---

## Results

All metrics are mean ± σ over 200 bootstrap resamples on the held-out test set (300,000 events).
AMS uses LHC-normalised weights scaled to the full 10 fb⁻¹ luminosity (s = 1,015 expected signal events, b = 1,050,370 background events — the Zenodo "LHC Events" counts, #15131565).

### Full experiment: 2M events · KAN hidden=[96,96,96], G=5, k=3

| Model | Params | AUC ± σ | AMS ± σ | Max AUC drop |
|---|---|---|---|---|
| XGBoost (baseline) | — | 0.8533 ± 0.0008 | 2.269 ± 0.017 | 0.052 |
| **KAN (base, deep)** | 190,465 | **0.8820 ± 0.0007** | 3.695 ± 0.171 | 0.101 |
| **KAN + grid extension** | 296,065 | 0.8799 ± 0.0007 | **3.935 ± 0.215** | 0.090 |
| **KAN-Adversarial** | 190,465 | 0.8743 ± 0.0007 | 2.949 ± 0.041 | **0.006** |

**A properly-tuned KAN beats XGBoost on both AUC and AMS.** The best significance (AMS) comes from KAN + grid extension. The deep KAN's extra capacity makes it more sensitive to energy-scale systematics, but **KAN-Adversarial** removes almost all of that sensitivity (Max AUC drop 0.006, ≈9× more robust than XGBoost) while still beating XGBoost on AUC and AMS — the best all-round model.

**Max AUC drop**: AUC degradation under momentum energy scale shifts γ ∈ [0.8, 1.2] — lower is more robust.

---

## References

- Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 (2024)
- Liu et al., *KAN 2.0*, arXiv:2408.10205 (2024)
- Louppe et al., *Learning to Pivot with Adversarial Networks*, NeurIPS 2017
- Cowan et al., *Asymptotic formulae for likelihood-based tests*, arXiv:1007.1727
- FAIR Universe HiggsML Dataset, Zenodo:10.5281/zenodo.15131565
