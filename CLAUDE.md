# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Main rules
- use `uv` to run python in this directory
- use `uvx ruff` to lint the files
- update the README.md file after each task completion
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

## Current Issues (correlation ~0.08, target 0.85+)

### Critical
1. **`influence.py` — Finite-diff JVP breaks CG**: `ggn_vector_product` uses `eps=1e-4` finite differences with `.data` mutation. Noise makes the linear operator inconsistent across CG iterations. **Fix**: Replace with exact JVP via `torch.func.jvp` + `functional_call`.
2. **`retrain.py` — PBRF double-penalizes parameters**: Proximal term iterates `model.named_parameters()` which includes both `convs` AND `lins` (mirror copies), doubling regularization and shrinking actual influence. **Fix**: Restrict optimizer + proximal to `sparse_params()` only.

### High
3. **`metrics.py` — `over_squashing` infeasible for `grad_A`**: 2708 forward passes per call; will OOM or take hours in dense-adj mode. **Fix**: Vectorize or skip `grad_A` for this metric.
4. **`metrics.py` — `dirichlet_energy` uses wrong edges in dense mode**: Sums energy over `data.edge_index` even when `adj` is provided. **Fix**: Derive `src, dst` from `adj.nonzero()` when adj is supplied.

### Medium
5. **`main.py` — PBRF hyperparams**: `PBRF_LR` and `PBRF_STEPS` may need tuning for convergence.
6. **`models.py` — `_sync_dense_from_sparse()`** uses `.data.copy_` inside `forward_dense`, which can break autograd. Wrap in `torch.no_grad()`.

## Task List
- [ ] Fix #1: Replace finite-diff JVP with `torch.func.jvp` in `ggn_vector_product`
- [ ] Fix #2: Restrict PBRF proximal + optimizer to `sparse_params()` only
- [ ] Fix #3: Make `over_squashing` feasible for dense-adj gradient (or skip `grad_A`)
- [ ] Fix #4: Fix `dirichlet_energy` edge set in dense mode
- [ ] Fix #5: Tune PBRF hyperparameters
- [ ] Fix #6: Wrap `_sync_dense_from_sparse` in `torch.no_grad()`
- [ ] Verify correlation reaches 0.85+ on 200 edges x 3 metrics
- [ ] Full run: `uv run main.py` produces `figure2.png`
