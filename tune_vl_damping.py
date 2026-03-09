"""Sweep VL damping + stochastic LiSSA batch sizes.

Usage: uv run tune_vl_damping.py
"""

import time
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from data import load_cora, sample_edges_for_deletion, sample_edges_for_insertion, edit_edge_index
from models import VanillaGCN, train_model
from metrics import validation_loss
from influence import compute_ihvp, compute_grad_A, compute_predicted_influence
from retrain import retrain_for_actual_influence, compute_actual_influence

HIDDEN_DIM = 64
NUM_LAYERS = 4
LR = 0.1
WEIGHT_DECAY = 1e-4
TRAIN_EPOCHS = 2000
NUM_EDGES = 200
SEED = 42
PBRF_LR = 0.01
PBRF_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sweep configs: (solver, damping, batch_size)
CONFIGS = [
    ("cg", 0.01, None),     # baseline
    ("lissa", 0.01, 10),    # stochastic LiSSA, small batch
    ("lissa", 0.01, 20),
    ("lissa", 0.01, 50),
    ("lissa", 0.01, None),  # exact LiSSA (same as CG)
    ("lissa", 0.1, 10),
    ("lissa", 0.1, 20),
]


def main():
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    print(f"Device: {DEVICE}")

    data = load_cora().to(DEVICE)
    num_classes = data.y.max().item() + 1
    model = VanillaGCN(data.num_features, HIDDEN_DIM, num_classes, num_layers=NUM_LAYERS).to(DEVICE)
    model = train_model(model, data, lr=LR, weight_decay=WEIGHT_DECAY, epochs=TRAIN_EPOCHS)

    del_edges = sample_edges_for_deletion(data, NUM_EDGES, seed=SEED)
    ins_edges = sample_edges_for_insertion(data, NUM_EDGES, seed=SEED)
    edges_meta = [(u, v, True) for u, v in del_edges] + [(u, v, False) for u, v in ins_edges]
    n_del = len(del_edges)

    # Compute actual influence once
    print(f"\n--- Computing actual influences (PBRF, {len(edges_meta)} edges) ---")
    actual_list = []
    for i, (u, v, is_del) in enumerate(edges_meta):
        ei_edited = edit_edge_index(data.edge_index, u, v, is_del)
        model_r = retrain_for_actual_influence(model, data, ei_edited, damping=0.01, lr=PBRF_LR, max_steps=PBRF_STEPS)
        actual, _, _ = compute_actual_influence(model, model_r, data, ei_edited, validation_loss, metric_kwargs={})
        actual_list.append(actual)
        if (i + 1) % 50 == 0:
            print(f"  Actual {i+1}/{len(edges_meta)}")
    actual_arr = np.array(actual_list)

    # Pre-compute grad_A (doesn't depend on solver/damping)
    grad_A = compute_grad_A(model, data, validation_loss, metric_kwargs={})

    # Sweep
    print(f"\n{'='*90}")
    print(f"{'Solver':<8} {'Damp':<6} {'Batch':<6} {'r_all':<8} {'r_del':<8} {'r_ins':<8} {'ρ_all':<8} {'Time':<8}")
    print(f"{'='*90}")

    for solver, damping, batch_size in CONFIGS:
        t0 = time.time()
        try:
            ihvp = compute_ihvp(model, data, validation_loss, metric_kwargs={},
                                damping=damping, solver=solver, max_iter=200,
                                batch_size=batch_size, verbose=False)

            pred_list = []
            for u, v, is_del in edges_meta:
                pred, _, _ = compute_predicted_influence(model, data, u, v, is_del, ihvp, grad_A)
                pred_list.append(pred)
            pred_arr = np.array(pred_list)

            r_all, _ = pearsonr(pred_arr, actual_arr)
            r_del, _ = pearsonr(pred_arr[:n_del], actual_arr[:n_del])
            r_ins, _ = pearsonr(pred_arr[n_del:], actual_arr[n_del:])
            rho_all, _ = spearmanr(pred_arr, actual_arr)
            elapsed = time.time() - t0

            bs_str = str(batch_size) if batch_size else "all"
            print(f"{solver:<8} {damping:<6.2f} {bs_str:<6} {r_all:<8.4f} {r_del:<8.4f} {r_ins:<8.4f} {rho_all:<8.4f} {elapsed:<8.1f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{solver:<8} {damping:<6.2f} {str(batch_size):<6} ERROR: {e} [{elapsed:.1f}s]")

    print("Done!")


if __name__ == "__main__":
    main()
