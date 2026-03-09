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

validation_loss correlation: **r = 0.9452** (20 edges, target 0.85+)
over_squashing and dirichlet_energy correlations still low — need further investigation.

## Resolved Issues

1. ~~**`influence.py` — Finite-diff JVP breaks CG**~~: Replaced with exact `torch.func.jvp` + `functional_call`.
2. ~~**`retrain.py` — PBRF double-penalizes parameters**~~: Restricted optimizer + proximal to `sparse_params()` only.
3. ~~**`metrics.py` — `over_squashing` infeasible for `grad_A`**~~: Samples 100 nodes instead of all 2708 (27x speedup), grad_A now enabled.
4. ~~**`metrics.py` — `dirichlet_energy` uses wrong edges in dense mode**~~: Dense mode now uses fully differentiable `(adj * sq_diff).sum() / adj.sum()` formulation.
5. ~~**`main.py` — PBRF hyperparams**~~: Tuned to `PBRF_LR=0.01`, `PBRF_STEPS=1000`.
6. ~~**`models.py` — `_sync_dense_from_sparse()` breaks autograd**~~: Wrapped in `torch.no_grad()`.

## Remaining Issues (from GPT-5.2 code review)

### Critical
1. **`forward_dense` does not exactly match `GCNConv`**: Tolerance is 0.05 (should be <1e-3). Mismatch corrupts `df/dA` gradients for both dirichlet_energy and over_squashing. **Fix**: Align dense normalization with PyG's `gcn_norm` exactly, then tighten `verify_dense_forward` tolerance.
2. **`over_squashing` neighbor masking is not differentiable w.r.t. `adj`**: `compute_L_hop_neighbors` uses discrete BFS on `edge_index`, so `df/dA` only captures gradient through `forward_dense`, not through neighborhood selection. **Fix**: Freeze neighbor structure from baseline graph and pass it explicitly so predicted and actual influence use consistent definitions.
3. **`dirichlet_energy` dense vs sparse formulations may diverge**: Dense uses `(adj * sq_diff).sum() / adj.sum()`, sparse uses `mean over edge_index`. These match only if `make_differentiable_adj` produces adjacency with same directionality/multiplicity as `edge_index`. **Fix**: Verify equivalence or align definitions explicitly.

### Medium
4. **`ggn_vector_product` VJP recomputes logits**: Step 3 calls `model(data.x, data.edge_index)` separately instead of reusing `functional_call` output from Step 1. Could desync if model has stateful behavior. **Fix**: Use `functional_call` for the VJP path too.
5. **`over_squashing` scaling doesn't account for skipped nodes**: Scale factor assumes all `sample_nodes` are used, but nodes with small neighborhoods are skipped. **Fix**: Track count of included nodes and scale by `num_nodes / count`.

### Config
- NUM_EDGES currently set to 20 for fast iteration; increase to 200 for final run

## Task List
- [x] Fix #1: Replace finite-diff JVP with `torch.func.jvp` in `ggn_vector_product`
- [x] Fix #2: Restrict PBRF proximal + optimizer to `sparse_params()` only
- [x] Fix #3: Make `over_squashing` feasible for dense-adj gradient (sample 100 nodes)
- [x] Fix #4: Fix `dirichlet_energy` edge set in dense mode
- [x] Fix #5: Tune PBRF hyperparameters
- [x] Fix #6: Wrap `_sync_dense_from_sparse` in `torch.no_grad()`
- [ ] Fix `forward_dense` to match `GCNConv` exactly, tighten tolerance to <1e-3
- [ ] Fix `over_squashing` neighbor masking consistency (freeze from baseline graph)
- [ ] Fix `dirichlet_energy` dense/sparse definition alignment
- [ ] Fix `ggn_vector_product` VJP to reuse `functional_call` output
- [ ] Fix `over_squashing` scaling to account for skipped nodes
- [ ] Verify correlation reaches 0.85+ on 200 edges x 3 metrics
- [x] Full run: `uv run main.py` produces `figure2.png`
