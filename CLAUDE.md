# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Main rules
- use `uv` to run python in this directory
- use `uvx ruff` to lint the files
- update the README.md file after each task completion
- update the CLAUDE.md file after each task completion with updates and current task list.
- if you run into any issues that you can't solve after 2 tries, use `/pal:chat` with `gemini-3-pro-preview`
- use `/pal:review` with `gpt-5.2` after each task completion

## Project Overview

Implementation of influence functions for edge edits in non-convex Graph Neural Networks (GNNs), based on the paper `2506.04694v2.pdf` included in the repo.

## Setup

- Python >=3.12 (pinned to 3.12 via `.python-version`)
- Uses **uv** as the package manager (`pyproject.toml`, no lock file yet)
- `uv sync` to install dependencies
- `uv run main.py` to run the main entry point

## Architecture

Reproducing Figure 2 from the paper: predicted vs actual influence on Cora with a 4-layer GCN.

| File | Role |
|------|------|
| `models.py` | VanillaGCN with sparse (`forward`) and dense (`forward_dense`) modes |
| `data.py` | Cora loading, edge sampling/editing utilities |
| `metrics.py` | Validation loss, over-squashing (Eq. 14), Dirichlet energy |
| `influence.py` | GGN-vector product, CG + LiSSA IHVP solvers, parameter shift (Eq. 10), message propagation (Eq. 11) |
| `retrain.py` | PBRF fine-tuning (Eq. 9) for computing actual influence |
| `plot.py` | 2x3 scatter plot (deletion/insertion rows, matching paper Figure 2 format) |
| `main.py` | End-to-end orchestration |
| `tune_vl_damping.py` | VL damping + solver sweep utility |
| `tune_hyperparams.py` | Weight decay + damping hyperparameter sweep |
| `sweep_checkpoint.py` | Training epoch checkpoint sweep for VL correlation |

## Current Status

- validation_loss correlation: **r = 0.836, ρ = 0.914** (200 edges, damping=1.0, L-BFGS PBRF)
- over_squashing correlation: **r = 0.925, ρ = 0.932** (200 edges, damping=0.1)
- dirichlet_energy correlation: **r = 0.906, ρ = 0.949** (200 edges, damping=0.1)
- Dense/sparse forward max diff: **5.72e-06** (well within 1e-3 tolerance)
- Per-metric damping implemented (paper tunes λ from {0.1, 0.01, 0.001, 0.0001})
- LiSSA solver implemented (both exact and stochastic mini-batch variants)
- L-BFGS PBRF optimizer (key fix for VL: SGD r=0.374 → L-BFGS r=0.836)
- Checkpoint sweep utility (`sweep_checkpoint.py`) for epoch/damping analysis
- Raw results saved to `results.pkl` for offline replotting
- Spearman rank correlation reported alongside Pearson

## VL Correlation Analysis

The paper achieves r=0.88 for VL on Cora (Figure 2). Our previous r=0.374 was fixed by two changes:

1. **L-BFGS for PBRF retraining** (dominant fix): SGD with lr=0.01 and 1000 steps did not converge precisely enough for the PBRF objective. L-BFGS with strong Wolfe line search (10 outer iterations, lr=1.0) converges to much tighter optima, giving accurate "actual influence" ground truth. This alone improved VL from r=0.374 to r=0.836.

2. **VL damping tuned to 1.0**: Higher damping (1.0 instead of 0.01) stabilizes the IHVP for VL's ill-conditioned GGN.

3. **Stochastic GGN scaling fix**: When using mini-batch GGN in LiSSA, the divisor was N (full train size) instead of batch_size, causing biased estimates. Fixed to divide by `len(batch_idx)`.

### Checkpoint sweep results (sweep_checkpoint.py)
- Epoch 100 + damping 5.0 peaks at r=0.969 (but val_acc only 68%)
- Epoch 2000 + damping 1.0 gives r=0.925 with proper val/test accuracy
- L-BFGS is the dominant fix; epoch selection is secondary

## Resolved Issues

1. ~~**`influence.py` — Finite-diff JVP breaks CG**~~: Replaced with exact `torch.func.jvp` + `functional_call`.
2. ~~**`retrain.py` — PBRF double-penalizes parameters**~~: Restricted optimizer + proximal to `sparse_params()` only.
3. ~~**`metrics.py` — `over_squashing` infeasible for `grad_A`**~~: Samples 100 nodes instead of all 2708 (27x speedup), grad_A now enabled.
4. ~~**`metrics.py` — `dirichlet_energy` uses wrong edges in dense mode**~~: Dense mode now uses fully differentiable formulation with fixed `edge_count` denominator.
5. ~~**`main.py` — PBRF hyperparams**~~: L-BFGS with `lr=1.0`, `max_steps=10` (was SGD `lr=0.01`, `steps=1000`).
6. ~~**`models.py` — `_sync_dense_from_sparse()` breaks autograd**~~: Wrapped in `torch.no_grad()`.
7. ~~**`models.py` — `forward_dense` wrong operation order**~~: Fixed to match GCNConv: `lin(x)` → `norm @ x` → `+ bias` (was `norm @ x` → `lin(x)`). Also set matmul precision to "highest".
8. ~~**`metrics.py` — `over_squashing` neighbor masking inconsistency**~~: Freeze baseline L-hop neighbors and pass explicitly via `L_hop_neighbors` kwarg.
9. ~~**`influence.py` — `ggn_vector_product` VJP path**~~: Now uses `functional_call` instead of `model(...)` for consistency.
10. ~~**`metrics.py` — `over_squashing` scaling for skipped nodes**~~: Track `included_count` and scale by `num_nodes / included_count`.
11. ~~**`dirichlet_energy` correlation low (r=0.11)**~~: Root cause was GGN near-singular with low damping, inflating IHVP and param_shift. Fixed with per-metric damping (DE λ=0.1 → r=0.947).
12. ~~**Dense DE O(N²C) allocation**~~: Replaced `logits.unsqueeze(0) - logits.unsqueeze(1)` with `||h_i||² + ||h_j||² - 2h_i^Th_j` identity.
13. ~~**VJP autograd.grad target mismatch**~~: Fixed to differentiate w.r.t. `param_dict.values()` instead of `params`.
14. ~~**train_model optimizer scope**~~: Changed to `model.sparse_params()` since dense params are overwritten by sync.
15. ~~**VL correlation r=0.374**~~: Root cause was SGD PBRF not converging precisely. Fixed with L-BFGS optimizer (r=0.374→0.836). Also fixed stochastic GGN scaling bug (divisor N→batch_size) and tuned VL damping to 1.0.
16. ~~**`influence.py` — stochastic GGN scaling**~~: Mini-batch GGN was divided by N instead of batch_size, giving biased estimates. One-line fix.

## Task List
- [x] Fix #1: Replace finite-diff JVP with `torch.func.jvp` in `ggn_vector_product`
- [x] Fix #2: Restrict PBRF proximal + optimizer to `sparse_params()` only
- [x] Fix #3: Make `over_squashing` feasible for dense-adj gradient (sample 100 nodes)
- [x] Fix #4: Fix `dirichlet_energy` edge set in dense mode
- [x] Fix #5: Tune PBRF hyperparameters
- [x] Fix #6: Wrap `_sync_dense_from_sparse` in `torch.no_grad()`
- [x] Fix `forward_dense` to match `GCNConv` exactly, tighten tolerance to <1e-3
- [x] Fix `over_squashing` neighbor masking consistency
- [x] Fix `dirichlet_energy` dense/sparse definition alignment
- [x] Fix `ggn_vector_product` VJP to reuse `functional_call` output
- [x] Fix `over_squashing` scaling to account for skipped nodes
- [x] Investigate dirichlet_energy low correlation (r = 0.11 → r = 0.947)
- [x] Implement per-metric damping (VL=0.01, OQ=0.1, DE=0.1)
- [x] Fix dense DE O(N²C) allocation
- [x] Fix VJP autograd.grad target
- [x] Fix train_model optimizer scope
- [x] Implement LiSSA solver (exact + stochastic)
- [x] Add Spearman rank correlation alongside Pearson
- [x] Outlier-robust plotting (percentile axis limits, 2x3 format matching paper)
- [x] VL damping sweep ({0.001..50.0}) — best r=0.72 at damping=5.0
- [x] Stochastic LiSSA sweep (batch_size={10,20,50}) — marginal improvement
- [x] Weight decay sweep ({1e-3..1e-5}) — WD=1e-3 helps VL but hurts OQ
- [x] Full run: `uv run main.py` produces `figure2.png`
- [x] L-BFGS PBRF optimizer (VL r=0.374→0.836)
- [x] Stochastic GGN scaling fix (divisor N→batch_size)
- [x] Checkpoint sweep (`sweep_checkpoint.py`) confirming overtraining hypothesis
- [x] VL damping tuned to 1.0 (was 0.01)
- [x] NaN guard in L-BFGS PBRF (falls back to SGD)
- [x] Raw results saved to `results.pkl`
- [ ] Remove `_sync_dense_from_sparse()` from `forward_dense()`, sync explicitly at call sites
- [ ] Replace identity-based param selection with name-based matching in `ggn_vector_product`
- [ ] Add NaN/negative-denominator guard in CG solver
