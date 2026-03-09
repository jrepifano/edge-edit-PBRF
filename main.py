"""Reproduce Figure 2 from the paper: Predicted vs Actual Influence on Cora.

Runs influence estimation for edge deletions and insertions on a 4-layer GCN
across 3 metrics: validation loss, over-squashing, and Dirichlet energy.
"""

import time

import torch

from data import (
    load_cora,
    sample_edges_for_deletion,
    sample_edges_for_insertion,
    edit_edge_index,
)
from models import VanillaGCN, train_model
from metrics import validation_loss, over_squashing, dirichlet_energy, compute_L_hop_neighbors
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
NUM_EDGES = 20  # per type (deletion/insertion) — use 200 for final run
SEED = 42
PBRF_LR = 0.01
PBRF_STEPS = 1000
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
    assert diff < 1e-3, f"Dense forward mismatch: {diff}"
    print("Dense forward verification PASSED")


def process_edges(model, data, edges, is_deletion, metric_fn, metric_kwargs,
                  ihvp, grad_A, label):
    """Process a list of edges (deletion or insertion), returning pred/actual lists."""
    predicted_list = []
    actual_list = []
    total = len(edges)
    t0 = time.time()

    print(f"\n--- Processing {total} {label} edges ---")
    for i, (u, v) in enumerate(edges):
        edge_t0 = time.time()

        # Predicted influence
        pred, ps, mp = compute_predicted_influence(
            model, data, u, v, is_deletion=is_deletion, ihvp=ihvp, grad_A=grad_A
        )

        # Actual influence via PBRF
        edge_index_edited = edit_edge_index(data.edge_index, u, v, is_deletion=is_deletion)
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

        elapsed = time.time() - edge_t0
        total_elapsed = time.time() - t0
        avg = total_elapsed / (i + 1)
        eta = avg * (total - i - 1)
        print(f"  [{label} {i+1}/{total}] edge=({u},{v}) "
              f"pred={pred:.4e} actual={actual:.4e} "
              f"(ps={ps:.4e} mp={mp:.4e}) "
              f"[{elapsed:.1f}s, ETA {eta:.0f}s]", flush=True)

    print(f"  {label} done in {time.time() - t0:.1f}s")
    return predicted_list, actual_list


def run_metric(model, data, metric_name, metric_fn, metric_kwargs,
               deletion_edges, insertion_edges):
    """Run influence estimation and actual influence for one metric."""
    metric_t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Processing metric: {metric_name}")
    print(f"{'='*60}")

    # Step 1: Compute IHVP (once per metric)
    print("\n--- Computing IHVP ---")
    t0 = time.time()
    ihvp = compute_ihvp(
        model, data, metric_fn, metric_kwargs=metric_kwargs,
        damping=DAMPING, cg_iter=CG_ITER, verbose=True,
    )
    print(f"IHVP computed in {time.time() - t0:.1f}s")

    # Step 2: Compute grad_A (once per metric)
    print("\n--- Computing grad_A ---")
    t0 = time.time()
    grad_A = compute_grad_A(model, data, metric_fn, metric_kwargs=metric_kwargs)
    print(f"||grad_A|| = {grad_A.norm().item():.6e}")
    print(f"grad_A computed in {time.time() - t0:.1f}s")

    # Step 3: Process edges
    del_pred, del_actual = process_edges(
        model, data, deletion_edges, is_deletion=True,
        metric_fn=metric_fn, metric_kwargs=metric_kwargs,
        ihvp=ihvp, grad_A=grad_A, label="deletion",
    )
    ins_pred, ins_actual = process_edges(
        model, data, insertion_edges, is_deletion=False,
        metric_fn=metric_fn, metric_kwargs=metric_kwargs,
        ihvp=ihvp, grad_A=grad_A, label="insertion",
    )

    print(f"\nMetric '{metric_name}' total time: {time.time() - metric_t0:.1f}s")

    return {
        "predicted": del_pred + ins_pred,
        "actual": del_actual + ins_actual,
        "is_deletion": [True] * len(del_pred) + [False] * len(ins_pred),
    }


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("highest")
    print(f"Device: {DEVICE}")
    print(f"Config: NUM_EDGES={NUM_EDGES}, PBRF_LR={PBRF_LR}, PBRF_STEPS={PBRF_STEPS}, "
          f"DAMPING={DAMPING}, CG_ITER={CG_ITER}")

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

    # --- Precompute baseline L-hop neighbors (frozen from baseline graph) ---
    print("\n--- Precomputing baseline L-hop neighbors ---")
    baseline_L_hop = compute_L_hop_neighbors(data.edge_index, data.num_nodes, NUM_LAYERS)
    print(f"Precomputed L-hop neighbors for {len(baseline_L_hop)} nodes")

    # --- Define metrics ---
    metrics = {
        "validation_loss": {
            "fn": validation_loss,
            "kwargs": {},
        },
        "over_squashing": {
            "fn": over_squashing,
            "kwargs": {"num_layers": NUM_LAYERS, "L_hop_neighbors": baseline_L_hop},
        },
        "dirichlet_energy": {
            "fn": dirichlet_energy,
            "kwargs": {"edge_count": data.edge_index.shape[1]},
        },
    }

    # --- Run all metrics ---
    results = {}
    total_t0 = time.time()
    for idx, (metric_name, metric_config) in enumerate(metrics.items()):
        print(f"\n>>> Metric {idx+1}/3: {metric_name}")
        results[metric_name] = run_metric(
            model, data, metric_name,
            metric_config["fn"], metric_config["kwargs"],
            deletion_edges, insertion_edges,
        )

        # Print running correlation after each metric
        from scipy.stats import pearsonr
        import numpy as np
        pred = np.array(results[metric_name]["predicted"])
        actual = np.array(results[metric_name]["actual"])
        if len(pred) > 2:
            corr, pval = pearsonr(pred, actual)
            print(f">>> {metric_name} correlation: r = {corr:.4f} (p = {pval:.2e})")

    print(f"\nTotal pipeline time: {time.time() - total_t0:.1f}s")

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
