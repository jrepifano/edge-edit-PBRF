"""Sweep weight_decay and VL damping to find best configuration.

Tests on 50 edges for speed. Paper tunes WD from {1e-3,...,1e-7}.
"""

import time
import torch
import numpy as np
from scipy.stats import pearsonr

from data import load_cora, sample_edges_for_deletion, sample_edges_for_insertion, edit_edge_index
from models import VanillaGCN, train_model
from metrics import validation_loss
from influence import compute_ihvp, compute_grad_A, compute_predicted_influence
from retrain import retrain_for_actual_influence, compute_actual_influence

HIDDEN_DIM = 64
NUM_LAYERS = 4
LR = 0.1
TRAIN_EPOCHS = 2000
NUM_EDGES = 50
SEED = 42
PBRF_LR = 0.01
PBRF_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT_DECAYS = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
VL_DAMPINGS = [0.01, 0.1, 1.0]


def eval_vl_correlation(model, data, del_edges, ins_edges, damping):
    """Compute VL predicted vs actual correlation."""
    edges_meta = [(u, v, True) for u, v in del_edges] + [(u, v, False) for u, v in ins_edges]

    # Actual influence (PBRF)
    actual_list = []
    for u, v, is_del in edges_meta:
        ei_edited = edit_edge_index(data.edge_index, u, v, is_del)
        model_r = retrain_for_actual_influence(model, data, ei_edited, damping=damping, lr=PBRF_LR, max_steps=PBRF_STEPS)
        actual, _, _ = compute_actual_influence(model, model_r, data, ei_edited, validation_loss, metric_kwargs={})
        actual_list.append(actual)

    # Predicted influence
    ihvp = compute_ihvp(model, data, validation_loss, metric_kwargs={}, damping=damping, solver="cg", max_iter=200, verbose=False)
    grad_A = compute_grad_A(model, data, validation_loss, metric_kwargs={})

    pred_list = []
    for u, v, is_del in edges_meta:
        pred, _, _ = compute_predicted_influence(model, data, u, v, is_del, ihvp, grad_A)
        pred_list.append(pred)

    pred_arr = np.array(pred_list)
    actual_arr = np.array(actual_list)

    # Separate del/ins correlations
    n_del = len(del_edges)
    r_del, _ = pearsonr(pred_arr[:n_del], actual_arr[:n_del]) if n_del > 2 else (float('nan'),)
    r_ins, _ = pearsonr(pred_arr[n_del:], actual_arr[n_del:]) if len(ins_edges) > 2 else (float('nan'),)
    r_all, _ = pearsonr(pred_arr, actual_arr)

    return r_all, r_del, r_ins


def main():
    torch.set_float32_matmul_precision("highest")
    print(f"Device: {DEVICE}")

    data = load_cora().to(DEVICE)
    num_classes = data.y.max().item() + 1
    del_edges = sample_edges_for_deletion(data, NUM_EDGES, seed=SEED)
    ins_edges = sample_edges_for_insertion(data, NUM_EDGES, seed=SEED)

    print(f"\n{'WD':<10} {'Damp':<8} {'ValAcc':<8} {'TestAcc':<8} {'r_all':<8} {'r_del':<8} {'r_ins':<8} {'Time':<8}")
    print("=" * 72)

    best_r = -1
    best_cfg = None

    for wd in WEIGHT_DECAYS:
        torch.manual_seed(SEED)
        model = VanillaGCN(data.num_features, HIDDEN_DIM, num_classes, num_layers=NUM_LAYERS).to(DEVICE)
        model = train_model(model, data, lr=LR, weight_decay=wd, epochs=TRAIN_EPOCHS)

        for damping in VL_DAMPINGS:
            t0 = time.time()
            try:
                r_all, r_del, r_ins = eval_vl_correlation(model, data, del_edges, ins_edges, damping)
                elapsed = time.time() - t0
                # Get accuracy from last training
                with torch.no_grad():
                    logits = model(data.x, data.edge_index)
                    val_acc = (logits[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()
                    test_acc = (logits[data.test_mask].argmax(1) == data.y[data.test_mask]).float().mean().item()
                print(f"{wd:<10.0e} {damping:<8.2f} {val_acc:<8.3f} {test_acc:<8.3f} {r_all:<8.4f} {r_del:<8.4f} {r_ins:<8.4f} {elapsed:<8.1f}")

                if r_all > best_r:
                    best_r = r_all
                    best_cfg = (wd, damping, val_acc, test_acc, r_all, r_del, r_ins)
            except Exception as e:
                print(f"{wd:<10.0e} {damping:<8.2f} ERROR: {e}")

    print(f"\n{'='*72}")
    if best_cfg:
        wd, d, va, ta, ra, rd, ri = best_cfg
        print(f"Best: WD={wd:.0e}, damping={d}, val={va:.3f}, test={ta:.3f}, "
              f"r_all={ra:.4f}, r_del={rd:.4f}, r_ins={ri:.4f}")


if __name__ == "__main__":
    main()
