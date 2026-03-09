# edge-edit-PBRF
Influence Functions for Edge Edits in Non-Convex Graph Neural Networks

Reproduces Figure 2 from [Heo et al. (2506.04694v2)](2506.04694v2.pdf): predicted vs actual influence of single edge edits on a 4-layer GCN trained on Cora, across 3 evaluation metrics (validation loss, over-squashing, Dirichlet energy).

## Setup

```bash
uv sync
uv run main.py
```

Requires Python >= 3.12. Uses `uv` as the package manager.

## Architecture

| File | Description |
|------|-------------|
| `models.py` | 4-layer VanillaGCN with sparse (`forward`) and dense (`forward_dense`) modes |
| `data.py` | Cora loading via PyG Planetoid, edge sampling and editing utilities |
| `metrics.py` | Three evaluation functions: validation loss, over-squashing (Eq. 14), Dirichlet energy |
| `influence.py` | GGN-vector product, CG-based IHVP, parameter shift (Eq. 10), message propagation (Eq. 11) |
| `retrain.py` | PBRF fine-tuning (Eq. 9) for computing actual influence |
| `plot.py` | Scatter plot generation (1x3 subplots, deletion/insertion markers, Pearson correlation) |
| `main.py` | End-to-end orchestration: train, sample edges, estimate + compute actual influence, plot |

## Method (Eq. 12)

The predicted influence of an edge edit decomposes into:
- **Parameter shift**: how changing the graph alters optimal model parameters, estimated via IHVP (G^{-1} grad_f) solved with conjugate gradient
- **Message propagation**: how changing the graph alters forward-pass computations, computed via df/dA on a dense adjacency matrix

Actual influence is computed by PBRF fine-tuning (Eq. 9) and evaluating metric difference.

## Configuration

Key hyperparameters in `main.py`:
- `HIDDEN_DIM=64`, `NUM_LAYERS=4`, `LR=0.1`, `WEIGHT_DECAY=1e-4`
- `DAMPING=0.01`, `CG_ITER=200`
- `NUM_EDGES=200` (per type), `PBRF_LR=0.01`, `PBRF_STEPS=1000`

## Status

- Model training: working (val acc ~79%)
- Dense/sparse forward equivalence: verified
- GGN-vector product: exact JVP via `torch.func.jvp` + `functional_call`
- CG solver, grad_A, PBRF retraining: implemented and working
- End-to-end pipeline: functional, produces `figure2.png`
- **validation_loss: r = 0.9452** (20 edges)
- over_squashing, dirichlet_energy: correlation tuning in progress
- Target: correlations 0.85+ across all 3 metrics
