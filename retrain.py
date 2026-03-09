import copy

import torch
import torch.nn.functional as F


def compute_bregman_divergence(logits_theta, logits_theta_s, labels):
    """Compute Bregman divergence D_L(h, h') for cross-entropy loss (Eq. 6).

    D_L(h, h', y) = L(h, y) - L(h', y) - grad_h' L(h', y)^T (h - h')
    """
    loss_theta = F.cross_entropy(logits_theta, labels, reduction="none")
    loss_theta_s = F.cross_entropy(logits_theta_s, labels, reduction="none")

    p_s = F.softmax(logits_theta_s, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits_theta.shape[1]).float()
    grad_h = p_s - one_hot

    diff_h = logits_theta - logits_theta_s.detach()
    linear_term = (grad_h.detach() * diff_h).sum(dim=1)

    bregman = loss_theta - loss_theta_s.detach() - linear_term
    return bregman.mean()


def compute_edge_edit_pbrf_loss(
    model, data, edge_index_orig, edge_index_edited,
    logits_theta_s_orig, theta_s_dict, damping, epsilon
):
    """Compute the edge-edit PBRF objective (Eq. 9).

    L = (1/N) sum_v D_L(h_v^{G,theta}, h_v^{G,theta_s})
        + (lambda/2) ||theta - theta_s||^2
        + epsilon * sum_v [L(h_v^{G,theta}) - L(h_v^{G_edited,theta})]

    Note: third term uses current theta on BOTH graphs (G and G_edited).
    """
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]

    # Forward with current params on original graph
    logits_current_orig = model(data.x, edge_index_orig)
    # Forward with current params on edited graph
    logits_current_edited = model(data.x, edge_index_edited)

    # Term 1: Bregman divergence on training nodes
    bregman = compute_bregman_divergence(
        logits_current_orig[train_idx],
        logits_theta_s_orig[train_idx],
        data.y[train_idx],
    )

    # Term 2: Proximal regularization
    proximal = 0.0
    for name, p in model.named_parameters():
        if name in theta_s_dict:
            proximal = proximal + ((p - theta_s_dict[name]) ** 2).sum()
    proximal = (damping / 2.0) * proximal

    # Term 3: Edge-edit response — uses current theta on both graphs
    loss_orig = F.cross_entropy(
        logits_current_orig[train_idx], data.y[train_idx], reduction="sum"
    )
    loss_edited = F.cross_entropy(
        logits_current_edited[train_idx], data.y[train_idx], reduction="sum"
    )
    edge_response = epsilon * (loss_orig - loss_edited)

    total = bregman + proximal + edge_response
    return total


def retrain_for_actual_influence(
    model,
    data,
    edge_index_edited,
    damping=0.01,
    lr=0.01,
    max_steps=1000,
    tol=1e-8,
    verbose=False,
):
    """Retrain model using PBRF fine-tuning (Eq. 9) to get theta*.

    Clones the model, optimizes the PBRF objective starting from theta_s.
    """
    model_retrained = copy.deepcopy(model)
    theta_s_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
    N = data.train_mask.sum().item()
    epsilon = -1.0 / N

    # Precompute logits at theta_s on original graph (constant reference)
    with torch.no_grad():
        logits_theta_s_orig = model(data.x, data.edge_index).detach()

    optimizer = torch.optim.SGD(model_retrained.parameters(), lr=lr)

    prev_loss = float("inf")
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = compute_edge_edit_pbrf_loss(
            model_retrained,
            data,
            data.edge_index,
            edge_index_edited,
            logits_theta_s_orig,
            theta_s_dict,
            damping,
            epsilon,
        )
        loss.backward()
        optimizer.step()

        # Sync dense params if model has them
        if hasattr(model_retrained, "_sync_dense_from_sparse"):
            model_retrained._sync_dense_from_sparse()

        loss_val = loss.item()
        if verbose and (step + 1) % 100 == 0:
            print(f"  PBRF step {step+1}: loss = {loss_val:.8f}")

        if abs(prev_loss - loss_val) < tol:
            if verbose:
                print(f"  PBRF converged at step {step+1}")
            break
        prev_loss = loss_val

    return model_retrained


def compute_actual_influence(model_orig, model_retrained, data, edge_index_edited,
                             metric_fn, metric_kwargs=None):
    """Compute actual influence: f(theta*, G_edited) - f(theta_s, G).

    theta* = retrained params, theta_s = original params.
    """
    if metric_kwargs is None:
        metric_kwargs = {}

    with torch.no_grad():
        kwargs_orig = dict(metric_kwargs)
        kwargs_orig["edge_index"] = data.edge_index
        f_orig = metric_fn(model_orig, data, **kwargs_orig).item()

        kwargs_edited = dict(metric_kwargs)
        kwargs_edited["edge_index"] = edge_index_edited
        f_edited = metric_fn(model_retrained, data, **kwargs_edited).item()

    return f_edited - f_orig
