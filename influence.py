import torch
import torch.nn.functional as F
from tqdm import tqdm


def _get_params(model):
    """Get the active parameters for influence computation (sparse conv params)."""
    if hasattr(model, "sparse_params"):
        return model.sparse_params()
    return [p for p in model.parameters() if p.requires_grad]


def ggn_vector_product(model, data, v_flat, damping=0.01, batch_idx=None):
    """Compute Generalized Gauss-Newton vector product: G @ v.

    G = (1/N) sum_{v in train} J_v^T H_v J_v + lambda*I

    Uses exact JVP via torch.func.jvp + functional_call.

    Args:
        batch_idx: Optional subset of training indices for stochastic GGN.
            If None, uses all training nodes (exact GGN).
            When provided, computes an unbiased estimate of the full-data
            (1/N) GGN-VP by still dividing by N (not |batch|).
    """
    params = _get_params(model)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    N = train_idx.shape[0]
    # Use mini-batch if provided (stochastic GGN for LiSSA)
    if batch_idx is not None:
        train_idx = batch_idx

    # Reshape v_flat into parameter shapes (tangent vectors)
    v_list = []
    offset = 0
    for p in params:
        numel = p.numel()
        v_list.append(v_flat[offset:offset + numel].view(p.shape))
        offset += numel

    # Build param dict and tangent dict for functional_call
    param_dict = {}
    tangent_dict = {}
    v_idx = 0
    for name, p in model.named_parameters():
        if any(p is sp for sp in params):
            param_dict[name] = p
            tangent_dict[name] = v_list[v_idx]
            v_idx += 1

    # Step 1: Exact JVP via torch.func.jvp + functional_call
    def fn(param_values):
        return torch.func.functional_call(model, param_values, (data.x, data.edge_index))

    logits_orig, Jv = torch.func.jvp(fn, (param_dict,), (tangent_dict,))

    # Step 2: Hessian-output product (softmax Hessian of cross-entropy)
    p = F.softmax(logits_orig[train_idx].detach(), dim=1)
    Jv_train = Jv[train_idx]
    pJv = (p * Jv_train).sum(dim=1, keepdim=True)
    HJv = p * Jv_train - p * pJv

    # Step 3: VJP: J^T @ HJv
    logits_for_grad = torch.func.functional_call(model, param_dict, (data.x, data.edge_index))
    grad_output = torch.zeros_like(logits_for_grad)
    grad_output[train_idx] = HJv.detach()

    vjp_targets = list(param_dict.values())
    grads = torch.autograd.grad(
        logits_for_grad, vjp_targets, grad_outputs=grad_output,
        retain_graph=False, allow_unused=True
    )
    result = torch.cat([
        (g if g is not None else torch.zeros_like(p_)).flatten()
        for g, p_ in zip(grads, vjp_targets)
    ])

    # Average over training set and add damping
    result = result / N + damping * v_flat
    return result


def estimate_lambda_max(model, data, damping=0.01, num_iters=50):
    """Estimate largest eigenvalue of G via power iteration."""
    params = _get_params(model)
    total_params = sum(p.numel() for p in params)
    device = params[0].device

    v = torch.randn(total_params, device=device)
    v = v / v.norm()

    for _ in range(num_iters):
        Gv = ggn_vector_product(model, data, v, damping=damping)
        lambda_est = v.dot(Gv)
        v = Gv / Gv.norm()

    return lambda_est.item()


def conjugate_gradient(model, data, target_vec, damping=0.01, max_iter=200,
                       tol=1e-6, verbose=True):
    """Solve G @ x = target_vec using conjugate gradient.

    CG converges in O(sqrt(kappa)) iterations for PD systems,
    much faster than LiSSA's O(kappa) for ill-conditioned problems.
    """
    x = torch.zeros_like(target_vec)
    r = target_vec.clone()  # r = b - Ax, and x=0 so r=b
    p = r.clone()
    rs_old = r.dot(r)
    residual = rs_old.sqrt().item()

    iterator = tqdm(range(max_iter), desc="CG") if verbose else range(max_iter)

    for k in iterator:
        Gp = ggn_vector_product(model, data, p, damping=damping)
        alpha = rs_old / (p.dot(Gp) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Gp
        rs_new = r.dot(r)

        residual = rs_new.sqrt().item()
        if verbose and (k + 1) % 20 == 0:
            tqdm.write(f"  CG iter {k+1}: residual = {residual:.2e}")

        if residual < tol:
            if verbose:
                print(f"CG converged at iteration {k+1}, residual = {residual:.2e}")
            break

        p = r + (rs_new / (rs_old + 1e-30)) * p
        rs_old = rs_new

    if verbose and residual >= tol:
        print(f"CG finished at {max_iter} iters, residual = {residual:.2e}")

    return x


def lissa(model, data, target_vec, damping=0.01, max_iter=10000, tol=1e-5,
          batch_size=None, verbose=True):
    """Solve G^{-1} @ v using stochastic LiSSA (Neumann series).

    Paper Section D, Eq. 37-38:
        r^(0) = v, r^(k+1) = v + (I - G_s) @ r^(k)
    where G_s = G/s, s > lambda_max(G), so G^{-1}v = (1/s) * G_s^{-1} * v.

    Uses stochastic mini-batch GGN estimates per iteration (as in the original
    LiSSA algorithm), providing implicit regularization via noise.

    Args:
        batch_size: Number of training nodes per mini-batch. None = all nodes (exact).
    """
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]

    # Estimate lambda_max for rescaling (use exact GGN for stability)
    lambda_max = estimate_lambda_max(model, data, damping=damping)
    s = lambda_max * 1.05  # safety margin
    if verbose:
        print(f"  LiSSA: lambda_max={lambda_max:.4e}, s={s:.4e}, "
              f"batch_size={batch_size or 'all'}")

    v_scaled = target_vec / s
    r = v_scaled.clone()
    diff = float("inf")

    iterator = tqdm(range(max_iter), desc="LiSSA") if verbose else range(max_iter)

    for k in iterator:
        # Stochastic mini-batch: sample training nodes for this iteration
        if batch_size is not None and batch_size < len(train_idx):
            perm = torch.randperm(len(train_idx), device=train_idx.device)[:batch_size]
            batch = train_idx[perm]
        else:
            batch = None  # use all training nodes

        Gr = ggn_vector_product(model, data, r, damping=damping, batch_idx=batch)
        r_new = v_scaled + r - Gr / s

        diff = (r_new - r).norm().item()
        if verbose and (k + 1) % 100 == 0:
            tqdm.write(f"  LiSSA iter {k+1}: delta = {diff:.2e}")

        if diff < tol:
            if verbose:
                print(f"LiSSA converged at iteration {k+1}, delta = {diff:.2e}")
            break

        r = r_new

    if verbose and diff >= tol:
        print(f"LiSSA finished at {max_iter} iters, delta = {diff:.2e}")

    return r


def compute_grad_f_theta(model, data, metric_fn, metric_kwargs=None):
    """Compute gradient of evaluation metric f w.r.t. model parameters.

    Returns flattened gradient vector over sparse params.
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    model.zero_grad()

    print(f"  compute_grad_f_theta: calling metric_fn={metric_fn.__name__}...", flush=True)
    val = metric_fn(model, data, **metric_kwargs)
    print(f"  compute_grad_f_theta: metric_fn returned {val.item():.6e}, computing grads...", flush=True)
    params = _get_params(model)
    grads = torch.autograd.grad(val, params, retain_graph=False, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).flatten()
        for g, p in zip(grads, params)
    ])


def compute_ihvp(model, data, metric_fn, metric_kwargs=None, damping=0.01,
                 solver="lissa", max_iter=200, batch_size=None, verbose=True):
    """Compute inverse-Hessian vector product: G^{-1} @ grad_f(theta_s).

    Computed once per metric, reused for all edges.

    Args:
        solver: "lissa" (paper default, stochastic Neumann series) or "cg" (conjugate gradient)
        max_iter: max iterations for the solver (200 for CG, 10000 for LiSSA default)
        batch_size: For LiSSA, number of training nodes per mini-batch (None=all).
    """
    if verbose:
        print("Computing grad_f(theta_s)...")
    grad_f = compute_grad_f_theta(model, data, metric_fn, metric_kwargs)
    if verbose:
        print(f"||grad_f|| = {grad_f.norm().item():.6e}")

    if solver == "lissa":
        lissa_iter = max_iter if max_iter != 200 else 10000
        if verbose:
            print(f"Running LiSSA to solve G^{{-1}} @ grad_f "
                  f"(max_iter={lissa_iter}, batch_size={batch_size or 'all'})...")
        ihvp = lissa(
            model, data, grad_f, damping=damping, max_iter=lissa_iter,
            batch_size=batch_size, verbose=verbose,
        )
    else:
        if verbose:
            print(f"Running CG to solve G^{{-1}} @ grad_f (max_iter={max_iter})...")
        ihvp = conjugate_gradient(
            model, data, grad_f, damping=damping, max_iter=max_iter, verbose=verbose
        )

    if verbose:
        print(f"||ihvp|| = {ihvp.norm().item():.6e}")
    return ihvp


def compute_grad_A(model, data, metric_fn, metric_kwargs=None):
    """Compute df/dA via dense backward pass.

    Returns (N, N) gradient matrix.
    """
    from data import make_differentiable_adj

    device = next(model.parameters()).device
    adj = make_differentiable_adj(data.edge_index, data.num_nodes).to(device)

    kwargs = dict(metric_kwargs) if metric_kwargs else {}
    kwargs["adj"] = adj
    kwargs.pop("edge_index", None)

    print(f"  compute_grad_A: calling metric_fn={metric_fn.__name__} with dense adj...", flush=True)
    val = metric_fn(model, data, **kwargs)
    print(f"  compute_grad_A: metric_fn returned {val.item():.6e}, computing grad_adj...", flush=True)
    grad_adj = torch.autograd.grad(val, adj, retain_graph=False)[0]
    return grad_adj.detach()


def compute_parameter_shift(model, data, edge_index_edited, ihvp):
    """Compute parameter shift term from Eq. 12.

    param_shift = (1/N) * ihvp^T @ (grad_theta_orig - grad_theta_edited)

    Positive sign because ε=-1/N cancels the negative from Eq. 10.
    """
    params = _get_params(model)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    N = train_idx.shape[0]

    # Gradient on original graph
    model.zero_grad()
    logits_orig = model(data.x, data.edge_index)
    loss_orig = F.cross_entropy(logits_orig[train_idx], data.y[train_idx], reduction="sum")
    grads_orig = torch.autograd.grad(loss_orig, params, retain_graph=False, allow_unused=True)
    grad_orig_flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).flatten()
        for g, p in zip(grads_orig, params)
    ])

    # Gradient on edited graph
    model.zero_grad()
    logits_edited = model(data.x, edge_index_edited)
    loss_edited = F.cross_entropy(
        logits_edited[train_idx], data.y[train_idx], reduction="sum"
    )
    grads_edited = torch.autograd.grad(loss_edited, params, retain_graph=False, allow_unused=True)
    grad_edited_flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).flatten()
        for g, p in zip(grads_edited, params)
    ])

    # Eq. 12: param_shift = (1/N) * ∇f^T G^{-1} Σ_v (∇L_orig - ∇L_edited)
    diff = grad_orig_flat - grad_edited_flat
    param_shift = (1.0 / N) * ihvp.dot(diff)
    return param_shift.item()


def compute_message_propagation(grad_A, u, v, is_deletion):
    """Compute message propagation term from Eq. 11.

    msg_prop = -(2*I[{u,v} in E] - 1) * (df/dA_uv + df/dA_vu)
    """
    indicator = 1.0 if is_deletion else 0.0
    sign = -(2.0 * indicator - 1.0)
    msg_prop = sign * (grad_A[u, v].item() + grad_A[v, u].item())
    return msg_prop


def compute_predicted_influence(model, data, u, v, is_deletion, ihvp, grad_A):
    """Compute predicted influence = parameter_shift + message_propagation (Eq. 12)."""
    from data import edit_edge_index

    edge_index_edited = edit_edge_index(data.edge_index, u, v, is_deletion)
    param_shift = compute_parameter_shift(model, data, edge_index_edited, ihvp)
    msg_prop = compute_message_propagation(grad_A, u, v, is_deletion)
    return param_shift + msg_prop, param_shift, msg_prop
