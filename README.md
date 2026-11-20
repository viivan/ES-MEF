# **Electrical-Guided Spatio-temporal Graph with Multimodal Evidence Fusion for Elevator Fault Diagnosis (ES-MEF)**

Source code for the electrical-guided spatio-temporal graph with multimodal evidence fusion method for elevator fault diagnosis.

![Python](https://img.shields.io/badge/python-3.13-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-ee4c2c)

## Overview

Elevator faults can produce concurrent electrical, vibration and acoustic responses. However, these signals differ markedly in their physical characteristics and sampling schemes, making it difficult for conventional feature-concatenation methods to exploit their complementary information. ES-MEF uses the electrical branch to guide the correction of vibration and acoustic features. It models within-modality temporal patterns with a local temporal graph and cross-modal relations with a heterogeneous global graph. State-aware evidence fusion then combines branch-level evidence to produce the final diagnosis.

The code includes the proposed method and scripts for component ablation, graph-structure and relation-edge ablations, modality-subset and missing-modality experiments, signal-degradation robustness tests, baseline comparisons and validation of the physical time delay.

## Features

- **Three-modality feature extraction**: LSTM for electrical signals, a multi-scale 1D CNN for vibration signals and a stacked 1D CNN for acoustic signals
- **Electrical-guided cross-modal feature correction（ECF）**: electrical features serve as queries in cross-attention, while a sample-specific gate adjusts the guidance strength when correcting vibration and acoustic features
- **Local temporal graph**: residual graph convolution aggregates features across temporal neighborhoods within each modality
- **Heterogeneous global graph**: relational graph convolution models synchronous, directed delay and intra-modal temporal relations
- **State-aware evidence fusion**: entropy-based discounting adjusts branch evidence, learnable class weights adjust class contributions and Dempster's rule combines the evidence
- **Experiment scripts**: component and graph-relation ablations, robustness tests, modality-subset and missing-modality experiments, and ten baselines evaluated under a common framework

## Requirements

- Python 3.13
- PyTorch 2.12.0
- numpy, scikit-learn, scipy, pandas

## Installation

```bash
git clone <repository-url>
cd ES-MEF
pip install -r requirements.txt
```

## Data

Place the data under `data/` at the project root, or point `ESMEF_ROOT` to the data root:

```bash
set ESMEF_ROOT=<project-root>
```

Layout:

```
<project-root>/
├─ data/
│  ├─ electrical/electrical_augmented.npz
│  ├─ vibration/
│  └─ acoustic/
├─ results/
└─ checkpoints/
```

## Usage

```bash
python experiments/run_main_esmef.py                 # proposed method (ES-MEF)
python experiments/run_ablation.py                   # component ablation
python experiments/run_graph_ablation.py             # graph structure and relation edges
python experiments/run_modality_subset.py            # single and dual modality
python experiments/run_missing_modality_ablation.py  # missing modality
python experiments/run_robustness_experiments.py     # signal degradation
python experiments/baselines_all.py --method all     # comparison baselines
python experiments/run_validate_physical_delay.py    # validation of the physical time delay
```

Results are written as JSON files under the corresponding `results/` subdirectory.

## Structure

```
.
├─ paths.py                     path resolution
├─ config.py                    hyperparameters
├─ es_mef.py                    main model
├─ subset_model.py              modality-subset model
├─ data_loader_multimodal.py    loading, pairing and temporal split
├─ models/
│  ├─ extractors.py             modality feature extractors
│  ├─ ecf.py                    ECF cross-modal correction
│  ├─ graph.py                  local GCN and global R-GCN
│  └─ fusion.py                 evidence fusion and classifiers
└─ experiments/
   ├─ run_main_esmef.py
   ├─ run_ablation.py
   ├─ run_graph_ablation.py
   ├─ run_modality_subset.py
   ├─ run_missing_modality_ablation.py
   ├─ run_robustness_experiments.py
   ├─ baselines_all.py
   └─ run_validate_physical_delay.py
```
