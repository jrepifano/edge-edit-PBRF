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
                   sample_nodes=100, seed=42):
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

    ei_for_neighbors = edge_index if edge_index is not None else data.edge_index
    L_hop = compute_L_hop_neighbors(ei_for_neighbors, data.num_nodes, num_layers)

    num_nodes = data.num_nodes

    # Sample a subset of nodes for tractability
    if sample_nodes is not None and sample_nodes < num_nodes:
        gen = torch.Generator().manual_seed(seed)
        node_indices = torch.randperm(num_nodes, generator=gen)[:sample_nodes].tolist()
        scale = num_nodes / sample_nodes
    else:
        node_indices = list(range(num_nodes))
        scale = 1.0

    total = torch.tensor(0.0, device=logits_orig.device)

    for v in node_indices:
        neighbor_set = L_hop[v]
        if len(neighbor_set) <= 1:
            continue
        x_modified = data.x.clone()
        neighbor_indices = [n for n in neighbor_set if n != v]
        if len(neighbor_indices) == 0:
            continue
        x_modified[neighbor_indices] = 0.0

        if adj is not None:
            logits_mod = model.forward_dense(x_modified, adj)
        else:
            logits_mod = model(x_modified, ei_for_neighbors)

        diff = logits_orig[v] - logits_mod[v]
        total = total + torch.norm(diff, p=2)

    return total * scale


def dirichlet_energy(model, data, edge_index=None, adj=None):
    """Dirichlet energy: mean ||h_u - h_v||^2 over edges.

    Measures over-smoothing — lower values mean more over-smoothing.
    """
    if adj is not None:
        logits = model.forward_dense(data.x, adj)
        # Fully differentiable w.r.t. adj: E = sum_ij A_ij * ||h_i - h_j||^2 / sum_ij A_ij
        diff = logits.unsqueeze(0) - logits.unsqueeze(1)  # (N, N, C)
        sq_diff = (diff * diff).sum(dim=2)  # (N, N)
        energy = (adj * sq_diff).sum() / (adj.sum() + 1e-10)
    else:
        ei = edge_index if edge_index is not None else data.edge_index
        logits = model(data.x, ei)
        src, dst = ei[0], ei[1]
        diff = logits[src] - logits[dst]
        energy = (diff * diff).sum(dim=1).mean()
    return energy
