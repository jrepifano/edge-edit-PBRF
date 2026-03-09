import torch
import torch.nn.functional as F


def validation_loss(model, data, edge_index=None, adj=None):
    """Mean cross-entropy on validation nodes. Returns differentiable scalar."""
    if adj is not None:
        logits = model.forward_dense(data.x, adj)
    else:
        ei = edge_index if edge_index is not None else data.edge_index
        logits = model(data.x, ei)
    return F.cross_entropy(logits[data.val_mask], data.y[data.val_mask])


def compute_L_hop_neighbors(edge_index, num_nodes, L):
    """Precompute L-hop neighbors for each node using BFS on adjacency.

    Returns dict: node_id -> set of node_ids reachable in exactly L hops.
    Actually returns nodes reachable in <= L hops (the L-hop neighborhood).
    """
    # Build adjacency list
    adj_list = [set() for _ in range(num_nodes)]
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        adj_list[ei[0, i]].add(ei[1, i])

    neighbors = {}
    for v in range(num_nodes):
        visited = {v}
        frontier = {v}
        for _ in range(L):
            next_frontier = set()
            for node in frontier:
                for nb in adj_list[node]:
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
        neighbors[v] = visited
    return neighbors


def over_squashing(model, data, num_layers, edge_index=None, adj=None,
                   sample_nodes=100, seed=42, L_hop_neighbors=None):
    """Over-squashing metric (Eq. 14).

    f_OQ = (N/S) * sum_{v in sample} ||h_v^G - h_v^{G'(v)}||_2

    where G'(v) zeros out features of L-hop neighbors of v.
    Samples `sample_nodes` nodes for tractability, scales result to full graph.
    """
    if adj is not None:
        logits_orig = model.forward_dense(data.x, adj)
    else:
        ei = edge_index if edge_index is not None else data.edge_index
        logits_orig = model(data.x, ei)

    if L_hop_neighbors is not None:
        L_hop = L_hop_neighbors
    else:
        ei_for_neighbors = edge_index if edge_index is not None else data.edge_index
        L_hop = compute_L_hop_neighbors(ei_for_neighbors, data.num_nodes, num_layers)

    num_nodes = data.num_nodes

    # Sample a subset of nodes for tractability
    if sample_nodes is not None and sample_nodes < num_nodes:
        gen = torch.Generator().manual_seed(seed)
        node_indices = torch.randperm(num_nodes, generator=gen)[:sample_nodes].tolist()
    else:
        node_indices = list(range(num_nodes))

    total = torch.tensor(0.0, device=logits_orig.device)
    included_count = 0

    ei_for_forward = edge_index if edge_index is not None else data.edge_index

    for v in node_indices:
        neighbor_set = L_hop[v]
        if len(neighbor_set) <= 1:
            continue
        neighbor_indices = [n for n in neighbor_set if n != v]
        if len(neighbor_indices) == 0:
            continue
        included_count += 1
        x_modified = data.x.clone()
        x_modified[neighbor_indices] = 0.0

        if adj is not None:
            logits_mod = model.forward_dense(x_modified, adj)
        else:
            logits_mod = model(x_modified, ei_for_forward)

        diff = logits_orig[v] - logits_mod[v]
        total = total + torch.norm(diff, p=2)

    return total * (num_nodes / max(included_count, 1))


def dirichlet_energy(model, data, edge_index=None, adj=None, edge_count=None):
    """Dirichlet energy: mean ||h_u - h_v||^2 over edges.

    Measures over-smoothing — lower values mean more over-smoothing.
    """
    if adj is not None:
        logits = model.forward_dense(data.x, adj)
        # Fully differentiable w.r.t. adj: E = sum_ij A_ij * ||h_i - h_j||^2 / count
        # Use identity ||h_i-h_j||^2 = ||h_i||^2 + ||h_j||^2 - 2*h_i^T*h_j
        # to avoid O(N^2*C) intermediate tensor
        sq_norms = (logits * logits).sum(dim=1)  # (N,)
        cross = logits @ logits.t()  # (N, N)
        sq_diff = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * cross  # (N, N)
        # Use fixed edge_count to avoid spurious gradient through denominator
        divisor = edge_count if edge_count is not None else (adj.sum() + 1e-10)
        energy = (adj * sq_diff).sum() / divisor
    else:
        ei = edge_index if edge_index is not None else data.edge_index
        logits = model(data.x, ei)
        src, dst = ei[0], ei[1]
        diff = logits[src] - logits[dst]
        count = edge_count if edge_count is not None else ei.shape[1]
        energy = (diff * diff).sum(dim=1).sum() / count
    return energy
