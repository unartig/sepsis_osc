# Combining Machine Learning and Dynamic Network Models for Sepsis Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13.13-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.10.0-orange.svg)](https://github.com/google/jax)

This repository contains the code accompanying the thesis:

> **Combining Machine-Learning and Dynamic Network Models to Improve Sepsis Prediction**
>
> [[Thesis PDF]](main_thesis.pdf) · [[(Outdated) Poster]](Poster.pdf) · [[DPG Talk]](presentation_dpg.pdf) · [[University Talk]](presentation_uni.pdf)

It provides two models:

**DNM — Dynamic Network Model** (renamed *Physiological Network Model* in the paper): a replica of the sepsis model described in [Berner et al. (2021)](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2021.730385/full) and [Sawicki et al. (2022)](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2022.904480/full), simulated via [JAX](https://github.com/google/jax) + [diffrax](https://github.com/patrick-kidger/diffrax).

**LDM — Latent Dynamics Model**: a deep-learning pipeline built on [equinox](https://github.com/patrick-kidger/equinox) that embeds the DNM parameter space to generate interpretable patient trajectories and enable online sepsis prediction.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Reproducing DNM Results](#reproducing-dnm-results)
- [Reproducing LDM Results](#reproducing-ldm-results)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Requirements

- **Python 3.12.12** (other 3.x versions may work but are untested)
- A CUDA-capable GPU is strongly recommended
- [RocksDB](https://github.com/facebook/rocksdb) (for persisting DNM simulations)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/unartig/sepsis_osc.git
cd sepsis_osc
```

### 2. Create a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` pins `jax[cuda13]`. If your CUDA version differs, replace with the appropriate extra, e.g. `jax[cuda12]`. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.
For CPU-only, use `jax[cpu]`.

### 4. Install the package (optional but recommended)

```bash
pip install -e .
```

---

## Reproducing DNM Results

The DNM sweeps over parameters β (beta) and σ (sigma) and runs GPU-accelerated ODE integration.

### Run a simulation

If you installed the package:
```bash
python -m sepsis_osc.dnm.dynamic_network_model
```

Or without installation:
```bash
PYTHONPATH=. python src/sepsis_osc/dnm/dynamic_network_model.py
```

### Persistent storage

To persist and checkpoint simulation results (useful for long parameter sweeps), install [RocksDB](https://github.com/facebook/rocksdb) before running.
This enables deduplication (skip already-computed parameter sets) and checkpointing (resume interrupted sweeps).

A pre-computed example storage **"data/DaisyFinal"** is included, covering two spaces:
- β ∈ (0, 1, step 0.01) × σ ∈ (0, 1.5, step 0.015)
- β ∈ (0.4, 0.7, step 0.003) × σ ∈ (0, 1.5, step 0.015)

---

## Reproducing LDM Results

The LDM requires access to the [MIMIC-IV](https://physionet.org/content/mimiciv/) clinical database and [YAIB](https://github.com/rvandewater/YAIB).
Access to MIMIC-IV requires credentialing via PhysioNet — see their [access instructions](https://physionet.org/content/mimiciv/2.2/).

### 1. Set up YAIB

Follow the [YAIB documentation](https://github.com/rvandewater/YAIB) to install.

### 2. Generate the cohort

Using the [YAIB-cohorts Docker container](https://github.com/rvandewater/YAIB-cohorts/tree/main/docker) is strongly recommended.
```bash
python misc/custom_sepsis_yaib_cohort.py
```

### 3. Train the LDM
Make sure `yaib_data_dir` in `src/sepsis_osc/utils/config.py` points to your local MIMIC-IV / YAIB data directory before running.

The YAIB PyTorch dataset is automatically converted to a JAX array and saved to `data/` on first run.
We are using a forked version of YAIB, since they do not allow to specify `train_size` when `complete_train=True`.

```bash
# With package installed:
python -m sepsis_osc.ldm.train_online
# or
python -m sepsis_osc.ldm.train_online_cv

# Without installation:
PYTHONPATH=. python src/sepsis_osc/ldm/train_online.py
# or
PYTHONPATH=. python src/sepsis_osc/ldm/train_online_cv.py
```

---

## Visualization

Visualization scripts are in `src/sepsis_osc/viz/`. The `viz_*.py` scripts can visualize:
- Ensemble systems or single instances of DNM initial value problems
- LDM predictions and patient trajectories

> **Note:** Some visualizations (e.g. parameter space plots) require `SystemMetrics` to be saved via the storage interface first.
See the `__main__` block in [`dynamic_network_model.py`](src/sepsis_osc/dnm/dynamic_network_model.py) for details.

Additional statistics and plots used in the thesis/paper can be reproduced using the notebooks in [`misc/`](misc/).

---

## Project Structure

```
sepsis_osc/
├── data/                   # Input / experimental datasets (not tracked in git)
├── figures/                # Stored visualizations (output)
├── misc/                   # Cohort creation and stats / visualization notebooks
├── src/
│   └── sepsis_osc/
│       ├── dnm/            # Dynamic Network Model (ODE simulation)
│       ├── ldm/            # Latent Dynamics Model (deep learning pipeline)
│       ├── utils/          # Utilities and config (edit config.py for paths)
│       ├── storage/        # FAISS + rocksdb persistent storage for DNM results
│       └── viz/            # Visualization functions
├── typst/                  # Typst source for thesis and figures
│   ├── chapters/           # Thesis text
│   ├── figures/            # Typst + CeTZ figures
│   └── images/             # SVGs and PNGs
├── main_thesis.pdf         # Full thesis
├── Poster.pdf              # (Outdated) Conference poster
├── presentation_dpg.pdf    # DPG talk slides
├── presentation_uni.pdf    # University talk slides
├── requirements.txt        # Python dependencies (Python 3.12.12)
└── pyproject.toml          # Package build config
```

---

## Citation

If you use this code or build on this work, please cite (bibtex of a _wip_ paper will be replacing the following):

```bibtex
@thesis{backes2026sepsis,
  title  = {Combining Machine-Learning and Dynamic Network Models to Improve Sepsis Prediction},
  author = {Juri Backes},
  year   = {2026},
  school = {Technical University Hamburg (TUHH)},
  url    = {https://github.com/unartig/sepsis_osc}
}
```

This work builds on:

```bibtex
@article{berner2021,
  title   = {Adaptive coupling of phase oscillators for sepsis modeling},
  author  = {Berner, R. and others},
  journal = {Frontiers in Network Physiology},
  year    = {2021},
  doi     = {10.3389/fnetp.2021.730385}
}

@article{sawicki2022,
  title   = {Modeling sepsis dynamics with coupled oscillators},
  author  = {Sawicki, J. and others},
  journal = {Frontiers in Network Physiology},
  year    = {2022},
  doi     = {10.3389/fnetp.2022.904480}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
