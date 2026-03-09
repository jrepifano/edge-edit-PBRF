"""Reproduce Figure 2 from the paper: Predicted vs Actual Influence on Cora.

Runs influence estimation for edge deletions and insertions on a 4-layer GCN
across 3 metrics: validation loss, over-squashing, and Dirichlet energy.
"""

import torch
from tqdm import tqdm

from data import (
    load_cora,
    sample_edges_for_deletion,
    sample_edges_for_insertion,
    edit_edge_index,
)
from models import VanillaGCN, train_model
from metrics import validation_loss, over_squashing, dirichlet_energy
from influence import (
    compute_ihvp,
    compute_grad_A,
    compute_predicted_influence,
)
from retrain import retrain_for_actual_influence, compute_actual_influence
from plot import plot_figure2

# --------------- Configuration ---------------
HIDDEN_DIM = 64
NUM_LAYERS = 4
LR = 0.1
WEIGHT_DECAY = 1e-4
TRAIN_EPOCHS = 2000
DAMPING = 0.01
CG_ITER = 200
NUM_EDGES = 200  # per type (deletion/insertion)
SEED = 42
PBRF_LR = 0.001
PBRF_STEPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------


def verify_dense_forward(model, data):
    """Verify that forward_dense matches forward within tolerance."""
    from data import make_differentiable_adj

    model.eval()
    with torch.no_grad():
        out_sparse = model(data.x, data.edge_index)
    adj = make_differentiable_adj(data.edge_index, data.num_nodes).to(DEVICE)
    with torch.no_grad():
        out_dense = model.forward_dense(data.x, adj)
    diff = (out_sparse - out_dense).abs().max().item()
    print(f"Dense vs Sparse forward max diff: {diff:.2e}")
    assert diff < 1e-4, f"Dense forward mismatch: {diff}"
    print("Dense forward verification PASSED")


def run_metric(model, data, metric_name, metric_fn, metric_kwargs,
               deletion_edges, insertion_edges):
    """Run influence estimation and actual influence for one metric."""
    print(f"\n{'='*60}")
    print(f"Processing metric: {metric_name}")
    print(f"{'='*60}")

    # Step 1: Compute IHVP (once per metric)
    print("\n--- Computing IHVP ---")
    ihvp = compute_ihvp(
        model, data, metric_fn, metric_kwargs=metric_kwargs,
        damping=DAMPING, cg_iter=CG_ITER, verbose=True,
    )

    # Step 2: Compute grad_A (once per metric)
    print("\n--- Computing grad_A ---")
    grad_A = compute_grad_A(model, data, metric_fn, metric_kwargs=metric_kwargs)
    print(f"||grad_A|| = {grad_A.norm().item():.6e}")

    predicted_list = []
    actual_list = []
    is_deletion_list = []

    # Step 3: Process deletion edges
    print(f"\n--- Processing {len(deletion_edges)} deletion edges ---")
    for i, (u, v) in enumerate(tqdm(deletion_edges, desc="Deletions")):
        # Predicted influence
        pred, _ps, _mp = compute_predicted_influence(
            model, data, u, v, is_deletion=True, ihvp=ihvp, grad_A=grad_A
        )

        # Actual influence via PBRF
        edge_index_edited = edit_edge_index(data.edge_index, u, v, is_deletion=True)
        model_retrained = retrain_for_actual_influence(
            model, data, edge_index_edited,
            damping=DAMPING, lr=PBRF_LR, max_steps=PBRF_STEPS,
        )
        actual = compute_actual_influence(
            model, model_retrained, data, edge_index_edited, metric_fn,
            metric_kwargs=metric_kwargs,
        )

        predicted_list.append(pred)
        actual_list.append(actual)
        is_deletion_list.append(True)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(deletion_edges)}] pred={pred:.6e}, actual={actual:.6e}")

    # Step 4: Process insertion edges
    print(f"\n--- Processing {len(insertion_edges)} insertion edges ---")
    for i, (u, v) in enumerate(tqdm(insertion_edges, desc="Insertions")):
        # Predicted influence
        pred, _ps, _mp = compute_predicted_influence(
            model, data, u, v, is_deletion=False, ihvp=ihvp, grad_A=grad_A
        )

        # Actual influence via PBRF
        edge_index_edited = edit_edge_index(data.edge_index, u, v, is_deletion=False)
        model_retrained = retrain_for_actual_influence(
            model, data, edge_index_edited,
            damping=DAMPING, lr=PBRF_LR, max_steps=PBRF_STEPS,
        )
        actual = compute_actual_influence(
            model, model_retrained, data, edge_index_edited, metric_fn,
            metric_kwargs=metric_kwargs,
        )

        predicted_list.append(pred)
        actual_list.append(actual)
        is_deletion_list.append(False)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(insertion_edges)}] pred={pred:.6e}, actual={actual:.6e}")

    return {
        "predicted": predicted_list,
        "actual": actual_list,
        "is_deletion": is_deletion_list,
    }


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    # --- Load Data ---
    print("\n--- Loading Cora ---")
    data = load_cora().to(DEVICE)
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, "
          f"Features: {data.num_features}, Classes: {data.y.max().item() + 1}")
    print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, "
          f"Test: {data.test_mask.sum()}")

    # --- Train GCN ---
    print("\n--- Training 4-layer GCN ---")
    num_classes = data.y.max().item() + 1
    model = VanillaGCN(
        data.num_features, HIDDEN_DIM, num_classes, num_layers=NUM_LAYERS
    ).to(DEVICE)
    model = train_model(
        model, data, lr=LR, weight_decay=WEIGHT_DECAY, epochs=TRAIN_EPOCHS
    )

    # --- Verify dense forward ---
    verify_dense_forward(model, data)

    # --- Sample edges ---
    print(f"\n--- Sampling {NUM_EDGES} edges for deletion and insertion ---")
    deletion_edges = sample_edges_for_deletion(data, NUM_EDGES, seed=SEED)
    insertion_edges = sample_edges_for_insertion(data, NUM_EDGES, seed=SEED)
    print(f"Sampled {len(deletion_edges)} deletion edges, {len(insertion_edges)} insertion edges")

    # --- Define metrics ---
    metrics = {
        "validation_loss": {
            "fn": validation_loss,
            "kwargs": {},
        },
        "over_squashing": {
            "fn": over_squashing,
            "kwargs": {"num_layers": NUM_LAYERS},
        },
        "dirichlet_energy": {
            "fn": dirichlet_energy,
            "kwargs": {},
        },
    }

    # --- Run all metrics ---
    results = {}
    for metric_name, metric_config in metrics.items():
        results[metric_name] = run_metric(
            model, data, metric_name,
            metric_config["fn"], metric_config["kwargs"],
            deletion_edges, insertion_edges,
        )

    # --- Plot ---
    print("\n--- Generating figure ---")
    plot_figure2(results, output_path="figure2.png")

    # --- Print correlations ---
    from scipy.stats import pearsonr
    import numpy as np

    print("\n--- Final Correlations ---")
    for metric_name in metrics:
        pred = np.array(results[metric_name]["predicted"])
        actual = np.array(results[metric_name]["actual"])
        corr, pval = pearsonr(pred, actual)
        print(f"{metric_name}: r = {corr:.4f} (p = {pval:.2e})")

    print("\nDone!")


if __name__ == "__main__":
    main()
