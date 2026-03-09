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


def over_squashing(model, data, num_layers, edge_index=None, adj=None, chunk_size=100):
    """Over-squashing metric (Eq. 14).

    f_OQ = sum_v ||h_v^G - h_v^{G'(v)}||_2

    where G'(v) zeros out features of L-hop neighbors of v.
    """
    if adj is not None:
        logits_orig = model.forward_dense(data.x, adj)
    else:
        ei = edge_index if edge_index is not None else data.edge_index
        logits_orig = model(data.x, ei)

    ei_for_neighbors = edge_index if edge_index is not None else data.edge_index
    L_hop = compute_L_hop_neighbors(ei_for_neighbors, data.num_nodes, num_layers)

    total = torch.tensor(0.0, device=logits_orig.device)
    num_nodes = data.num_nodes

    # Process in chunks to manage memory
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        for v in range(start, end):
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

    return total


def dirichlet_energy(model, data, edge_index=None, adj=None):
    """Dirichlet energy: mean ||h_u - h_v||^2 over edges.

    Measures over-smoothing — lower values mean more over-smoothing.
    """
    if adj is not None:
        logits = model.forward_dense(data.x, adj)
    else:
        ei = edge_index if edge_index is not None else data.edge_index
        logits = model(data.x, ei)

    ei = edge_index if edge_index is not None else data.edge_index
    src, dst = ei[0], ei[1]
    diff = logits[src] - logits[dst]
    energy = (diff * diff).sum(dim=1).mean()
    return energy
