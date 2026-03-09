# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Main rules
- use `uv` to run python in this directory
- use `uvx ruff` to lint the files
- update the README.md file after each task completion
- update the CLAUDE.md file after each task completion with updates and current task list.
- if you run into any issues that you can't solve after 2 tries, use `/pal:chat` with `gemini-3-pro`
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
| `influence.py` | GGN-vector product, CG-based IHVP, parameter shift (Eq. 10), message propagation (Eq. 11) |
| `retrain.py` | PBRF fine-tuning (Eq. 9) for computing actual influence |
| `plot.py` | Scatter plot generation |
| `main.py` | End-to-end orchestration |

## Current Status

- validation_loss correlation: **r = 0.9809** (20 edges)
- over_squashing correlation: **r = 0.8711** (20 edges)
- dirichlet_energy correlation: **r = 0.1120** (20 edges) — still low, needs investigation
- Dense/sparse forward max diff: **5.72e-06** (well within 1e-3 tolerance)

## Resolved Issues

1. ~~**`influence.py` — Finite-diff JVP breaks CG**~~: Replaced with exact `torch.func.jvp` + `functional_call`.
2. ~~**`retrain.py` — PBRF double-penalizes parameters**~~: Restricted optimizer + proximal to `sparse_params()` only.
3. ~~**`metrics.py` — `over_squashing` infeasible for `grad_A`**~~: Samples 100 nodes instead of all 2708 (27x speedup), grad_A now enabled.
4. ~~**`metrics.py` — `dirichlet_energy` uses wrong edges in dense mode**~~: Dense mode now uses fully differentiable formulation with fixed `edge_count` denominator.
5. ~~**`main.py` — PBRF hyperparams**~~: Tuned to `PBRF_LR=0.01`, `PBRF_STEPS=1000`.
6. ~~**`models.py` — `_sync_dense_from_sparse()` breaks autograd**~~: Wrapped in `torch.no_grad()`.
7. ~~**`models.py` — `forward_dense` wrong operation order**~~: Fixed to match GCNConv: `lin(x)` → `norm @ x` → `+ bias` (was `norm @ x` → `lin(x)`). Also set matmul precision to "highest".
8. ~~**`metrics.py` — `over_squashing` neighbor masking inconsistency**~~: Freeze baseline L-hop neighbors and pass explicitly via `L_hop_neighbors` kwarg.
9. ~~**`influence.py` — `ggn_vector_product` VJP path**~~: Now uses `functional_call` instead of `model(...)` for consistency.
10. ~~**`metrics.py` — `over_squashing` scaling for skipped nodes**~~: Track `included_count` and scale by `num_nodes / included_count`.

## Remaining Issues

### Critical
1. **`dirichlet_energy` correlation still low (r = 0.11)**: Root cause (confirmed by GPT-5.2 review): PBRF retrains on cross-entropy to find `theta*`, but actual influence evaluates Dirichlet energy at `theta*`. The parameter shift component reflects how `theta` changes to minimize CE, not how DE changes. **Potential fix**: Decompose actual influence into structure-only (`f(theta_s, G_edited) - f(theta_s, G)`) vs parameter-update (`f(theta*, G_edited) - f(theta_s, G_edited)`) to isolate where correlation breaks.

### High
2. **Dense Dirichlet energy O(N^2*C) allocation** (`metrics.py:108-110`): `logits.unsqueeze(0) - logits.unsqueeze(1)` creates (N,N,C) tensor. Use `||h_i||^2 + ||h_j||^2 - 2h_i^Th_j` identity to avoid.
3. **VJP `autograd.grad` target mismatch** (`influence.py:55-66`): Should differentiate w.r.t. `param_dict.values()` not `params` for full consistency with `functional_call`.

### Medium
4. **`train_model` optimizes all params** (`models.py:83`): Should use `model.sparse_params()` since dense params are overwritten by `_sync_dense_from_sparse` every step.

### Config
- NUM_EDGES currently set to 20 for fast iteration; increase to 200 for final run

## Task List
- [x] Fix #1: Replace finite-diff JVP with `torch.func.jvp` in `ggn_vector_product`
- [x] Fix #2: Restrict PBRF proximal + optimizer to `sparse_params()` only
- [x] Fix #3: Make `over_squashing` feasible for dense-adj gradient (sample 100 nodes)
- [x] Fix #4: Fix `dirichlet_energy` edge set in dense mode
- [x] Fix #5: Tune PBRF hyperparameters
- [x] Fix #6: Wrap `_sync_dense_from_sparse` in `torch.no_grad()`
- [x] Fix `forward_dense` to match `GCNConv` exactly, tighten tolerance to <1e-3 (was wrong op order: now lin→norm→bias; also set matmul precision to "highest" to avoid TF32 drift)
- [x] Fix `over_squashing` neighbor masking consistency (freeze baseline L-hop neighbors, pass via kwarg)
- [x] Fix `dirichlet_energy` dense/sparse definition alignment (fixed count in both paths, fixed dense denominator to use `edge_count`)
- [x] Fix `ggn_vector_product` VJP to reuse `functional_call` output
- [x] Fix `over_squashing` scaling to account for skipped nodes (track `included_count`)
- [ ] Investigate dirichlet_energy low correlation (r = 0.11)
- [ ] Verify correlation reaches 0.85+ on 200 edges x 3 metrics
- [x] Full run: `uv run main.py` produces `figure2.png`
