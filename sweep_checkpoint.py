"""Sweep training checkpoints to find optimal epoch for VL correlation.

Tests the hypothesis that overtraining (100% train acc) causes GGN ill-conditioning,
which breaks the first-order Taylor approximation for validation loss.

Trains once, saves checkpoints, evaluates VL correlation at each.
"""

import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from data import (
    edit_edge_index,
    load_cora,
    sample_edges_for_deletion,
    sample_edges_for_insertion,
)
from influence import compute_grad_A, compute_ihvp, compute_predicted_influence
from metrics import (
    compute_L_hop_neighbors,
    dirichlet_energy,
    over_squashing,
    validation_loss,
)
from models import VanillaGCN, train_model
from retrain import compute_actual_influence, retrain_for_actual_influence

HIDDEN_DIM = 64
NUM_LAYERS = 4
LR = 0.1
WEIGHT_DECAY = 1e-4
TRAIN_EPOCHS = 2000
NUM_EDGES = 25  # per type (25 del + 25 ins = 50 total)
SEED = 42
PBRF_LR = 1.0
PBRF_STEPS = 10
PBRF_OPTIMIZER = "lbfgs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_EPOCHS = [100, 200, 300, 500, 700, 1000, 1500, 2000]
VL_DAMPINGS = [0.01, 0.1, 1.0, 5.0]


def eval_correlation(model, data, del_edges, ins_edges, metric_fn, metric_kwargs,
                     damping, optimizer_type="lbfgs", pbrf_lr=1.0, pbrf_steps=10):
    """Compute predicted vs actual correlation for a given metric."""
    edges_meta = [(u, v, True) for u, v in del_edges] + [
        (u, v, False) for u, v in ins_edges
    ]

    # Predicted influence (IHVP + grad_A, computed once)
    ihvp = compute_ihvp(
        model, data, metric_fn, metric_kwargs=metric_kwargs,
        damping=damping, solver="cg", max_iter=200, verbose=False,
    )
    grad_A = compute_grad_A(model, data, metric_fn, metric_kwargs=metric_kwargs)

    pred_list = []
    actual_list = []
    for u, v, is_del in edges_meta:
        pred, _, _ = compute_predicted_influence(
            model, data, u, v, is_del, ihvp, grad_A
        )
        pred_list.append(pred)

        ei_edited = edit_edge_index(data.edge_index, u, v, is_del)
        model_r = retrain_for_actual_influence(
            model, data, ei_edited, damping=damping,
            lr=pbrf_lr, max_steps=pbrf_steps,
            optimizer_type=optimizer_type,
        )
        actual, _, _ = compute_actual_influence(
            model, model_r, data, ei_edited, metric_fn, metric_kwargs=metric_kwargs,
        )
        actual_list.append(actual)

    pred_arr = np.array(pred_list)
    actual_arr = np.array(actual_list)
    r, _ = pearsonr(pred_arr, actual_arr)
    rho, _ = spearmanr(pred_arr, actual_arr)
    return r, rho


def main():
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint epochs: {CHECKPOINT_EPOCHS}")
    print(f"VL dampings: {VL_DAMPINGS}")
    print(f"PBRF: optimizer={PBRF_OPTIMIZER}, lr={PBRF_LR}, steps={PBRF_STEPS}")

    data = load_cora().to(DEVICE)
    num_classes = data.y.max().item() + 1

    del_edges = sample_edges_for_deletion(data, NUM_EDGES, seed=SEED)
    ins_edges = sample_edges_for_insertion(data, NUM_EDGES, seed=SEED)
    print(f"Edges: {len(del_edges)} del + {len(ins_edges)} ins = {len(del_edges) + len(ins_edges)} total")

    # Train once with checkpoints
    print("\n--- Training with checkpoints ---")
    model = VanillaGCN(
        data.num_features, HIDDEN_DIM, num_classes, num_layers=NUM_LAYERS
    ).to(DEVICE)
    _, checkpoints = train_model(
        model, data, lr=LR, weight_decay=WEIGHT_DECAY, epochs=TRAIN_EPOCHS,
        checkpoint_epochs=CHECKPOINT_EPOCHS,
    )

    # Print checkpoint summary
    print(f"\n{'Epoch':<8} {'TrainAcc':<10} {'ValAcc':<10} {'Confidence':<12} {'Loss':<10}")
    print("=" * 50)
    for ep in CHECKPOINT_EPOCHS:
        ck = checkpoints[ep]
        print(f"{ep:<8} {ck['train_acc']:<10.4f} {ck['val_acc']:<10.4f} "
              f"{ck['confidence']:<12.4f} {ck['train_loss']:<10.6f}")

    # Evaluate VL correlation at each checkpoint × damping
    print(f"\n{'Epoch':<8} {'Damp':<8} {'r':<10} {'rho':<10} {'Time':<8}")
    print("=" * 50)

    best_r = -1
    best_cfg = None

    for ep in CHECKPOINT_EPOCHS:
        # Load checkpoint into a fresh model
        ck_model = VanillaGCN(
            data.num_features, HIDDEN_DIM, num_classes, num_layers=NUM_LAYERS
        ).to(DEVICE)
        ck_model.load_state_dict(checkpoints[ep]["state_dict"])
        ck_model.eval()

        for damping in VL_DAMPINGS:
            t0 = time.time()
            try:
                r, rho = eval_correlation(
                    ck_model, data, del_edges, ins_edges,
                    validation_loss, {},
                    damping=damping,
                    optimizer_type=PBRF_OPTIMIZER,
                    pbrf_lr=PBRF_LR,
                    pbrf_steps=PBRF_STEPS,
                )
                elapsed = time.time() - t0
                print(f"{ep:<8} {damping:<8.2f} {r:<10.4f} {rho:<10.4f} {elapsed:<8.1f}s")

                if r > best_r:
                    best_r = r
                    best_cfg = (ep, damping, r, rho)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"{ep:<8} {damping:<8.2f} ERROR: {e} [{elapsed:.1f}s]")

    print(f"\n{'='*50}")
    if best_cfg:
        ep, d, r, rho = best_cfg
        ck = checkpoints[ep]
        print(f"Best VL: epoch={ep}, damping={d}, r={r:.4f}, rho={rho:.4f}")
        print(f"  train_acc={ck['train_acc']:.4f}, val_acc={ck['val_acc']:.4f}, "
              f"confidence={ck['confidence']:.4f}")

        # Evaluate OQ and DE at best checkpoint to check for regression
        print(f"\n--- Evaluating OQ and DE at epoch={ep} ---")
        ck_model = VanillaGCN(
            data.num_features, HIDDEN_DIM, num_classes, num_layers=NUM_LAYERS
        ).to(DEVICE)
        ck_model.load_state_dict(checkpoints[ep]["state_dict"])
        ck_model.eval()

        baseline_L_hop = compute_L_hop_neighbors(
            data.edge_index, data.num_nodes, NUM_LAYERS
        )

        for metric_name, metric_fn, metric_kwargs, metric_damping in [
            ("over_squashing", over_squashing,
             {"num_layers": NUM_LAYERS, "L_hop_neighbors": baseline_L_hop}, 0.1),
            ("dirichlet_energy", dirichlet_energy,
             {"edge_count": data.edge_index.shape[1]}, 0.1),
        ]:
            t0 = time.time()
            r, rho = eval_correlation(
                ck_model, data, del_edges, ins_edges,
                metric_fn, metric_kwargs,
                damping=metric_damping,
                optimizer_type=PBRF_OPTIMIZER,
                pbrf_lr=PBRF_LR,
                pbrf_steps=PBRF_STEPS,
            )
            elapsed = time.time() - t0
            print(f"  {metric_name}: r={r:.4f}, rho={rho:.4f} [{elapsed:.1f}s]")


if __name__ == "__main__":
    main()
