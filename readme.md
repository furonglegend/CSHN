# README.md

```markdown
# Hyperspherical Uncertainty Decomposition with Causal Identification

This repository provides the official PyTorch implementation of **Our Method**,  
a hyperspherical uncertainty decomposition framework with causal structure identification via information geometry.

Our method models epistemic uncertainty on a unit hypersphere using von Mises–Fisher (vMF) geometry, decomposes predictive uncertainty into epistemic and aleatoric components, and integrates structural causal modeling for robust and interpretable learning.

---

## 🔍 Overview

Modern predictive systems often suffer from:

- Overconfident predictions
- Poor calibration under distribution shift
- Lack of interpretability
- Entangled uncertainty sources

**Our method addresses these challenges by:**

1. Representing features on a unit hypersphere
2. Modeling epistemic uncertainty through vMF concentration
3. Learning aleatoric variance explicitly
4. Performing uncertainty-aware fusion
5. Incorporating causal structure learning
6. Enforcing entropy calibration and causal regularization

---

## 📂 Project Structure

```text
.
├── data/
│   ├── prepare_snare.py
│   ├── prepare_pheme.py
│   ├── prepare_amigos.py
│   ├── loaders.py
│   ├── dataset.py
│   ├── temporal_hypergraph.py
│   ├── snare_dataset.py
│   ├── pheme_dataset.py
│   └── amigos_dataset.py
│
├── models/
│   ├── spherical_encoder.py
│   ├── vmf_head.py
│   ├── aleatoric_head.py
│   ├── fusion_head.py
│   ├── spherical_mp.py
│   ├── causal_scm.py
│   ├── regularizers.py
│   ├── loss.py
│   └── sphunc_model.py
│
├── training/
│   ├── trainer.py
│   ├── callbacks.py
│   ├── optim.py
│   └── logger.py
│
├── evaluation/
│   ├── calibration.py
│   ├── causal_recovery.py
│   └── intervention.py
│
├── utils/
│   ├── io.py
│   ├── logging.py
│   ├── seed.py
│   ├── metrics.py
│   ├── plotting.py
│   └── helpers.py
│
├── experiments/
│   ├── run_snare.py
│   ├── run_pheme.py
│   ├── run_amigos.py
│   └── grid_search.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
│
├── requirements.txt
└── README.md

````

