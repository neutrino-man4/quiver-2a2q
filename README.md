# 2A2Q: Two-Atom–Two-Qubit Molecular Embedding for HOMO-LUMO Gap Regression

**Authors:** Aritra Bal, Michael Binder, Markus Klute, Benedikt Maier, Michael Spannowsky

**Contact:** [aritra.bal@kit.edu](mailto:aritra.bal@kit.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2606.02785-b31b1b.svg)](https://arxiv.org/abs/2606.02785)
[![ICML 2026](https://img.shields.io/badge/ICML_2026-AI4Physics_Workshop-purple.svg)](https://arxiv.org/abs/2606.02785)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://etpwww.etp.kit.edu/~abal/projects/quiver/)

---

## Overview

This repository contains the implementation of the **2A2Q** (Two-Atom–Two-Qubit) variational quantum circuit (VQC), introduced as part of the [QUIVER](https://arxiv.org/abs/2606.02785) framework. The 2A2Q circuit encodes molecular structure into a quantum state and is trained to regress the **HOMO-LUMO gap** Δε = ε_HOMO − ε_LUMO on the [QM9](https://www.nature.com/articles/sdata201422) dataset.

The trained 2A2Q circuit is used within QUIVER to extract the **Quantum Fisher Information Matrix (QFIM)** — a geometry-aware, basis-independent summary of higher-order correlations captured by the learned quantum state manifold. This QFIM serves as a complementary quantum view that is fused into a classical graph neural network (DimeNet++) to improve molecular property prediction.

---

## The 2A2Q Encoding

Each molecule is represented as a 10-qubit system, with one qubit assigned to each heavy atom (unused qubit slots are filled with randomly sampled hydrogen atoms).

**Per-atom initialization.** Each qubit *j* is initialized with a species-dependent rotation:

```
RY(w^j_atom) |0>
```

where `w^j_atom` is a trainable parameter encoding the atomic species occupying qubit *j*.

**Pairwise entanglement.** For every pair of atoms *(i, j)* satisfying `d_ij < d_CUTOFF = 1.7 Å` and connected by a chemical bond, a two-qubit entanglement block is applied. The encoding angles are:

```
ω1^(ij) = e_d1 · (1 − d_ij / d_CUTOFF) · cos(θ_ij)
ω2^(ij) = e^(ij)_bond · π
ω3^(ij) = e_d2 · (1 − d_ij / d_CUTOFF) · cos(φ_ij)
```

where `e_d1`, `e_d2` are learnable distance-scaling parameters, `e^(ij)_bond` is a learnable bond-type entanglement parameter, and `d_ij`, `θ_ij`, `φ_ij` are the pairwise distance, zenith angle, and azimuthal angle for the atom pair. The pairwise entanglement unitary applied to the pair is:

```
U_ij = ( I_YY(ω3) I_ZZ(ω2) I_XX(ω1) ) ( RY(w^i_atom) ⊗ RY(w^j_atom) ) |00>
```

where `I_XX`, `I_YY`, `I_ZZ` are Ising-type two-qubit interactions.

**Trainable rotations.** After each entanglement stage, a per-qubit trainable rotation sequence `RZ · RY · RZ` is applied with independent parameters per qubit. The above constitutes one circuit layer; N = 2 layers are stacked in the final architecture.

**Measurement.** The HOMO-LUMO gap prediction is extracted from the observable:

```
H = Σ c_i Z_i   (i = 1, ..., N)
```

where `{c_i}` are trainable coefficients and `Z_i` is the Pauli-Z operator on qubit *i*. Since the gap is strictly positive, the raw expectation value is shifted by Σ|c_i|. The circuit is optimized using the Huber loss.

**QFIM output.** The resulting QFIM is a 10 × 10 grid of 6 × 6 sub-blocks (a 60 × 60 real symmetric matrix), where the off-diagonal sub-block Q_ij captures the coherent coupling between atoms *i* and *j* through the intrinsic geometry of the quantum state manifold. This matrix is consumed downstream by QDimeNet++ as described in the QUIVER paper.

All circuit simulations use [PennyLane](https://pennylane.ai/).

---

## Repository Structure

```
bioqinn/
├── configs/
│   ├── YAML/
│   │   ├── qm9.yaml           # User-facing hyperparameter configuration
│   │   └── base.yaml
│   ├── configuration.py       # Config loader: merges defaults + YAML overrides
│   └── defaults.py            # Default parameter values
├── data_handlers/
│   ├── qm9_dataloader.py      # PyG-based dense dataloader
│   └── qm9_h5_dataloader.py   # HDF5-backed dataloader (used in training)
├── data_processors/
│   └── h5_maker_qm9.py        # One-time HDF5 dataset creation script
├── quantum/
│   ├── architectures.py       # 2A2Q variational quantum circuit definition
│   └── trainer.py             # Training loop, validation, LR decay, checkpointing
├── requirements.txt
└── train.py                   # Main training entrypoint
```

---

## Dependencies

```
pennylane_lightning_gpu==0.44.0
torch-geometric==2.7.0
h5py==3.15.1
loguru==0.7.3
PyYAML==6.0.3
rdkit==2025.9.6
```

See [requirements.txt](requirements.txt) for the full list.

---

## Usage

### Step 1: Prepare the Dataset (run once)

The `h5_maker_qm9.py` script downloads the QM9 dataset via PyTorch Geometric, filters molecules by heavy-atom count, computes pairwise geometric edge features (bond type, polar angle θ, azimuthal angle φ, interatomic distance), and writes train/val/test splits to HDF5 files.

```bash
python3 data_processors/h5_maker_qm9.py
```

This must be run once before training. Output paths are configured in the script via `SAVE_ROOT`. The default split filters molecules with 5–9 heavy atoms (50/10/40 train/val/test).

**HDF5 schema per split:**

| Dataset          | Shape                            | Description                          |
|------------------|----------------------------------|--------------------------------------|
| `node_features`  | `(N, MAX_NODES, 9)`              | Atomic features + atom count scalars |
| `edge_features`  | `(N, MAX_NODES, MAX_NODES, 4)`   | Bond type, θ, φ, distance (Å)        |
| `targets`        | `(N, 19)`                        | QM9 molecular properties             |
| `n_atoms`        | `(N, 2)`                         | Total and heavy atom counts          |

---

### Step 2: Configure the Run

Default hyperparameters are in [configs/defaults.py](configs/defaults.py). Override any parameter via a YAML file. An annotated example is at [configs/YAML/qm9.yaml](configs/YAML/qm9.yaml).

Key configuration fields:

```yaml
setup:
  run_id: "run_001"       # Identifier for the output directory
  epochs: 50
  batch_size: 1           # Quantum circuit executes one sample at a time
  train_n: 5000           # Molecules seen by the VQC during training
  val_n: 1000

model:
  n_qubits: 10            # One qubit per heavy atom (up to 10)
  num_layers: 2           # Circuit depth (N = 2 in the paper)
  operations_per_layer: 3 # RZ-RY-RZ per qubit per layer
  device: "lightning.qubit"
  backend: "autograd"

optimizer:
  name: "adam"
  lr: 0.01
  lr_decay: true

loss:
  name: "huber"           # Huber loss as described in the paper

paths:
  train: "/path/to/qm9_train.h5"
  val:   "/path/to/qm9_val.h5"
  test:  "/path/to/qm9_test.h5"
  model_dir: "/path/to/save/models"
```

---

### Step 3: Train

```bash
python3 train.py --config configs/YAML/qm9.yaml
```

Outputs are written to `<model_dir>/<run_id>/`:

| File | Description |
|------|-------------|
| `config.yaml` | Fully resolved run configuration |
| `logs/train.log` | Training log |
| `circuit.png` | Circuit diagram (drawn at epoch 0) |
| `checkpoints/weights_epoch_NNN.npy` | Per-epoch weight snapshots |
| `trained_model/weights_best.npy` | Weights at best validation MAE |
| `trained_model/weights_final.npy` | Weights at end of training |
| `predictions.png` | Validation diagnostics |

---

## Citation

If you use this code or the 2A2Q embedding in your work, please cite:

```bibtex
@inproceedings{bal2026quiver,
  title     = {{QUIVER}: {QU}antum-{I}nformed {V}iews for {E}nhanced {R}epresentations in Large Machine Learning Models},
  author    = {Bal, Aritra and Binder, Michael and Klute, Markus and Maier, Benedikt and Spannowsky, Michael},
  booktitle = {ICML 2026 Workshop on AI for Physics},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.02785}
}
```

The 2A2Q encoding builds on the 1P1Q particle embedding developed for jet physics:

```bibtex
@article{bal2025,
  title   = {One particle - one qubit: Particle physics data encoding for quantum machine learning},
  author  = {Bal, Aritra and Klute, Markus and Maier, Benedikt and Oughton, Melik and Pezone, Eric and Spannowsky, Michael},
  journal = {Phys. Rev. D},
  volume  = {112},
  pages   = {076004},
  year    = {2025},
  doi     = {10.1103/l8y2-87vq},
  url     = {https://link.aps.org/doi/10.1103/l8y2-87vq}
}
```
