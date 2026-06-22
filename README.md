# Exploring Multi-Order Markov Chains for Road Network Mobility Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20code-6f42c1)
![Paper](https://img.shields.io/badge/venue-ATMOS%202026-lightgrey)

Associated code for the paper *Exploring Multi-Order Markov Chains for Road Network Mobility Prediction* (ATMOS 2026). We study next-node prediction on urban road networks using a family of multi-order Markov chain models of increasing complexity, and benchmark them against established sequential predictors on graph-mapped versions of the **WorldMove** dataset.

## Abstract

Accurate mobility prediction is a key component of intelligent transportation systems, enabling applications such as traffic management, route planning, and personalised mobility services. Existing methods follow two paradigms: individual models, which capture user-specific behaviour but suffer from data scarcity, and collective models, which exploit population-level patterns but lack online adaptability. We explore a family of multi-order Markov Chain models of increasing complexity, from a simple max-order baseline (MC) to an entropy-weighted variant (HW-MC), up to the Dual-Memory Markov Chain (DM-MC), which combines a population-based memory with an online individual memory fused through entropy-based weighting. We further introduce a dual evaluation protocol assessing next-node prediction, where ground-truth updates are continuously available, and auto-regressive forecasting, where trajectories are generated from a seed without real-time input. Under this protocol, MC achieves the highest next-node accuracy and the deepest auto-regressive horizon, while DM-MC matches competitive baselines (AKOM, CPT+, MOGen, IOHMM) in accuracy and assigns substantially higher confidence (NLL) to predicted nodes, with the lowest endpoint distance across all models. We additionally release graph-mapped versions of the WorldMove dataset as an open benchmark for road-topology-aware mobility modelling.

## Models

We explore three Markov chain models, each adding a mechanism on top of the previous one, evaluated against four baseline families.

| Model | Description |
| --- | --- |
| **MC** | Max-order Markov chain with greedy longest-context back-off. |
| **HW-MC** | Entropy-weighted mixture of all observed orders (down-weights low-support orders). |
| **DM-MC** | Dual-Memory Markov Chain: a *Population Memory* (PM) built offline plus an *Individual Memory* (IM) updated online, fused by entropy weighting that shifts reliance toward the IM as personal evidence accumulates. |

**Baselines:** AKOM, CPT+, MOGen (reimplemented for a fair comparison on road-network trajectories) and IOHMM (run via its public implementation).

All Markov models are evaluated at orders `k ∈ {2, 5, 10}` over a 5-fold cross-validation setup on four road networks.

## Datasets

Trajectories come from **WorldMove**, a synthetic mobility dataset that mimics the statistical properties of daily human movement while preserving privacy by construction. For each city, the corresponding road network is retrieved from **OpenStreetMap (OSM)** and each raw trajectory is mapped onto it, so every trip becomes a sequence of OSM node identifiers.

| Dataset | Nodes | Edges | Trips | Users | Days | Files |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Abu Dhabi (AE) | 53,251 | 112,591 | 18,676 | 500 | 15 | `data/worldmove_609_AE.csv` + `data/worldmove_609_AE_network.json` |
| Frankfurt (DE) | 59,219 | 139,436 | 34,148 | 500 | 15 | `data/worldmove_431_DE.csv` + `data/worldmove_431_DE_network.json` |
| New York (US) | 128,785 | 331,874 | 142,383 | 500 | 15 | `data/worldmove_380_US.csv` + `data/worldmove_380_US_network.json` |
| Singapore (SG) | 43,941 | 91,939 | 76,421 | 500 | 15 | `data/worldmove_539_SG.csv` + `data/worldmove_539_SG_network.json` |

## Evaluation protocol

The dual protocol assesses two complementary operational contexts:

- **Next-node prediction** — continuous access to ground truth. Given the previous 10 nodes, predict the next node. Metrics: **Accuracy** (top-1 correctness) and **NLL** (mean negative log-probability in bits assigned to the true next node; lower means more confident and better calibrated).
- **Auto-regressive forecasting** — closed-loop rollout from a 10-node seed with no further real-time input; the model's own predictions are fed back as context, so errors accumulate. Metrics: **Node Horizon Prediction** (mean number of consecutive nodes predicted while cumulative top-1 accuracy stays ≥ 75%) and **Endpoint distance** (km between predicted and ground-truth final node).

## Quickstart

### 1) Environment

```bash
python -m venv .venv
```

- Windows (PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3) Train and evaluate (single command)

`train_and_evaluate.py` splits trajectories once at the trajectory level (train 60% / val 20% / test 20%, fixed seed, no leakage), trains every model and baseline, evaluates them on the shared test set, and writes results + plots:

```bash
python train_and_evaluate.py \
    --data-path  data/worldmove_380_US.csv \
    --graph-path data/worldmove_380_US_network.json \
    --output-dir results_unified
```

Outputs in `results_unified/`:

- `unified_results.json` — all numbers
- `comparison_table.csv` — spreadsheet-ready
- `comparison_table.tex` — LaTeX-ready
- `comparison.png` — comparison bar chart

**Skip baselines** for a quick Markov-only run:

```bash
python train_and_evaluate.py --skip-baselines
```

**Custom options** (sensible defaults provided):

```bash
python train_and_evaluate.py \
    --orders 2 5 10 \
    --n-splits 5 \
    --test-ratio 0.20 \
    --val-ratio  0.20 \
    --seed 42
```

### 4) Separate training and evaluation

To train once and score repeatedly, split the two phases. `train_models.py` fits every model across the requested splits and saves them to disk:

```bash
python train_models.py \
    --data-path  data/worldmove_380_US.csv \
    --graph-path data/worldmove_380_US_network.json \
    --models-dir saved_models
```

`evaluate_models.py` then loads the saved models and computes both next-node and auto-regressive metrics:

```bash
python evaluate_models.py \
    --models-dir saved_models \
    --output-dir results_unified
```

Use `--no-autoregressive` to skip the (slower) closed-loop evaluation, or `--splits 0 1` to restrict to specific folds.

## Results

Averaged over the four WorldMove cities. Across all models, accuracy improves with Markov order. DM-MC matches the accuracy of the best baselines while assigning **substantially lower NLL**, concentrating far more probability mass on the correct node. In auto-regressive rollout, plain higher-order chains reach the deepest node horizon, while DM-MC10 achieves the lowest endpoint distance.

| Model | Accuracy ↑ | NLL (bits) ↓ | Node horizon (75%) ↑ | Endpoint dist. (km) ↓ |
| --- | ---: | ---: | ---: | ---: |
| DM-MC2 | 0.9315 | 0.3947 | 13.26 | 7.2543 |
| DM-MC5 | 0.9437 | 0.3052 | 14.62 | 6.7841 |
| DM-MC10 | 0.9477 | **0.2924** | 14.82 | **5.0340** |
| AKOM | 0.8844 | 1.3130 | 13.23 | 5.2928 |
| CPT+ | 0.9439 | 1.5978 | 14.15 | 6.2676 |
| IOHMM | 0.9329 | 2.6111 | 12.96 | 7.3373 |
| MOGen | 0.9142 | 1.1285 | 15.60 | 5.3783 |
| HW-MC10 | 0.9486 | 2.3122 | 15.31 | 6.1721 |
| MC10 | **0.9490** | 10.7262 | **18.06** | 5.9047 |

## Repository layout

```
train_and_evaluate.py   unified train + evaluate + plots
train_models.py         train all models/splits, save to disk
evaluate_models.py      load saved models, score next-node + auto-regressive
models/                 MC/HW-MC (simple_markov.py), DM-MC (graph_idyom.py),
                        baselines (akom, cpt, mogen, iohmm)
data/                   WorldMove CSV trajectories + OSM network JSON graphs
plot_results.py         comparison figures
results_unified/        generated results and tables
```

> Note: some internal module/identifier names predate the paper's terminology.
> In the code, `simple_markov.py` corresponds to MC/HW-MC and the dual-memory model in `graph_idyom.py` corresponds to DM-MC (PM/IM ≙ LTM/STM).

## Citation

If you use this repository in your research, please cite the paper.

```bibtex
@inproceedings{baudru2026multiorder,
    title     = {Exploring Multi-Order Markov Chains for Road Network Mobility Prediction},
    author    = {Baudru, Julien and Bono Rossell\'o, Lluc and Bersini, Hugues},
    booktitle = {26th Symposium on Algorithmic Approaches for Transportation Modelling, Optimization, and Systems (ATMOS)},
    year      = {2026}
}
```
