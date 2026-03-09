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
| `influence.py` | GGN-vector product, CG + LiSSA IHVP solvers, parameter shift (Eq. 10), message propagation (Eq. 11) |
| `retrain.py` | PBRF fine-tuning (Eq. 9) for computing actual influence |
| `plot.py` | 2x3 scatter plots (deletion/insertion rows) with percentile axis limits, matching paper format |
| `main.py` | End-to-end orchestration: train, sample edges, estimate + compute actual influence, plot |
| `tune_vl_damping.py` | VL-specific damping + solver sweep utility |
| `tune_hyperparams.py` | Weight decay + damping hyperparameter sweep |

## Method (Eq. 12)

The predicted influence of an edge edit decomposes into:
- **Parameter shift**: how changing the graph alters optimal model parameters, estimated via IHVP (G^{-1} grad_f) solved with LiSSA (paper default) or conjugate gradient
- **Message propagation**: how changing the graph alters forward-pass computations, computed via df/dA on a dense adjacency matrix

Actual influence is computed by PBRF fine-tuning (Eq. 9) and evaluating metric difference.

## IHVP Solvers

- **LiSSA** (default): Neumann series approximation (Section D, Eq. 37-38) with optional stochastic mini-batch GGN estimates. Paper uses LiSSA with 10,000 iterations.
- **CG**: Conjugate gradient. Converges faster for well-conditioned systems. Both solvers produce identical results when using full-batch GGN.

## Configuration

Key hyperparameters in `main.py`:
- `HIDDEN_DIM=64`, `NUM_LAYERS=4`, `LR=0.1`, `WEIGHT_DECAY=1e-4`
- `SOLVER="lissa"`, `CG_ITER=200`, `NUM_EDGES=200` (per type), `PBRF_LR=0.01`, `PBRF_STEPS=1000`
- Per-metric damping (paper tunes λ from {0.1, 0.01, 0.001, 0.0001}):
  - validation_loss: λ=0.01
  - over_squashing: λ=0.1
  - dirichlet_energy: λ=0.1

## Status

- Model training: working (val acc ~79%)
- Dense/sparse forward equivalence: verified (max diff 5.72e-06)
- GGN-vector product: exact JVP via `torch.func.jvp` + `functional_call`
- LiSSA + CG solvers, grad_A, PBRF retraining: implemented and working
- End-to-end pipeline: functional, produces `figure2.png`
- Reports both Pearson (r) and Spearman (ρ) correlations

### Correlations (200 edges)

| Metric | Damping | Pearson r | Spearman ρ | Paper r |
|--------|---------|-----------|------------|---------|
| validation_loss | 0.01 | 0.374 | 0.633 | 0.88 |
| **over_squashing** | 0.1 | **0.914** | 0.878 | 0.94 |
| **dirichlet_energy** | 0.1 | **0.947** | 0.865 | 0.95 |

VL gap analysis: The paper tunes LR, hidden_dim, and weight_decay on a grid search. Our fixed hyperparameters produce an over-trained model (100% train acc) with ill-conditioned GGN, causing param_shift outliers that degrade VL Pearson correlation. See CLAUDE.md for full analysis.
