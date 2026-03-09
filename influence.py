import torch
import torch.nn.functional as F
from tqdm import tqdm


def _get_params(model):
    """Get the active parameters for influence computation (sparse conv params)."""
    if hasattr(model, "sparse_params"):
        return model.sparse_params()
    return [p for p in model.parameters() if p.requires_grad]


def ggn_vector_product(model, data, v_flat, damping=0.01):
    """Compute Generalized Gauss-Newton vector product: G @ v.

    G = (1/N) sum_{v in train} J_v^T H_v J_v + lambda*I

    Uses JVP (finite diff) -> Hessian-output product -> VJP decomposition.
    """
    params = _get_params(model)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    N = train_idx.shape[0]

    # Reshape v_flat into parameter shapes
    v_list = []
    offset = 0
    for p in params:
        numel = p.numel()
        v_list.append(v_flat[offset : offset + numel].view(p.shape))
        offset += numel

    # Step 1: JVP via finite differences
    eps = 1e-4
    orig_params = [p.data.clone() for p in params]
    for p, vp in zip(params, v_list):
        p.data.add_(vp, alpha=eps)
    with torch.no_grad():
        logits_plus = model(data.x, data.edge_index)
    for p, op in zip(params, orig_params):
        p.data.copy_(op)
    with torch.no_grad():
        logits_orig = model(data.x, data.edge_index)

    Jv = (logits_plus - logits_orig) / eps  # (N_nodes, C)

    # Step 2: Hessian-output product for cross-entropy on training nodes
    # H_h L = diag(p) - p p^T, so H @ x = p*x - p*(p^T x)
    p = F.softmax(logits_orig[train_idx], dim=1)
    Jv_train = Jv[train_idx]
    pJv = (p * Jv_train).sum(dim=1, keepdim=True)
    HJv = p * Jv_train - p * pJv

    # Step 3: VJP: J^T @ HJv
    logits_for_grad = model(data.x, data.edge_index)
    grad_output = torch.zeros_like(logits_for_grad)
    grad_output[train_idx] = HJv

    grads = torch.autograd.grad(
        logits_for_grad, params, grad_outputs=grad_output,
        retain_graph=False, allow_unused=True
    )
    result = torch.cat([
        (g if g is not None else torch.zeros_like(p_)).flatten()
        for g, p_ in zip(grads, params)
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


def compute_grad_f_theta(model, data, metric_fn, metric_kwargs=None):
    """Compute gradient of evaluation metric f w.r.t. model parameters.

    Returns flattened gradient vector over sparse params.
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    model.zero_grad()

    val = metric_fn(model, data, **metric_kwargs)
    params = _get_params(model)
    grads = torch.autograd.grad(val, params, retain_graph=False, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).flatten()
        for g, p in zip(grads, params)
    ])


def compute_ihvp(model, data, metric_fn, metric_kwargs=None, damping=0.01,
                 cg_iter=200, verbose=True):
    """Compute inverse-Hessian vector product: G^{-1} @ grad_f(theta_s).

    Uses conjugate gradient. Computed once per metric, reused for all edges.
    """
    if verbose:
        print("Computing grad_f(theta_s)...")
    grad_f = compute_grad_f_theta(model, data, metric_fn, metric_kwargs)
    if verbose:
        print(f"||grad_f|| = {grad_f.norm().item():.6e}")

    if verbose:
        print("Running CG to solve G^{-1} @ grad_f...")
    ihvp = conjugate_gradient(
        model, data, grad_f, damping=damping, max_iter=cg_iter, verbose=verbose
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

    val = metric_fn(model, data, **kwargs)
    grad_adj = torch.autograd.grad(val, adj, retain_graph=False)[0]
    return grad_adj.detach()


def compute_parameter_shift(model, data, edge_index_edited, ihvp):
    """Compute parameter shift term from Eq. 10.

    param_shift = -(1/N) * ihvp^T @ (grad_theta_orig - grad_theta_edited)
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
