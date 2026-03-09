import torch
import numpy as np
from torch_geometric.datasets import Planetoid


def load_cora(root="./data"):
    """Load Cora dataset with standard Yang et al. splits."""
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    return data


def sample_edges_for_deletion(data, n, seed=42):
    """Sample n existing undirected edges for deletion.

    Returns list of (u, v) tuples with u < v.
    """
    rng = np.random.RandomState(seed)
    edge_index = data.edge_index.cpu().numpy()
    # Get unique undirected edges (u < v)
    mask = edge_index[0] < edge_index[1]
    edges = list(zip(edge_index[0][mask], edge_index[1][mask]))
    indices = rng.choice(len(edges), size=min(n, len(edges)), replace=False)
    return [edges[i] for i in indices]


def sample_edges_for_insertion(data, n, seed=42):
    """Sample n non-edges for insertion.

    Returns list of (u, v) tuples with u < v.
    """
    rng = np.random.RandomState(seed)
    num_nodes = data.num_nodes
    edge_index = data.edge_index.cpu().numpy()
    edge_set = set(zip(edge_index[0], edge_index[1]))

    non_edges = []
    attempts = 0
    max_attempts = n * 100
    while len(non_edges) < n and attempts < max_attempts:
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u == v:
            attempts += 1
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set and (v, u) not in edge_set:
            non_edges.append((u, v))
            edge_set.add((u, v))
            edge_set.add((v, u))
        attempts += 1
    return non_edges


def edit_edge_index(edge_index, u, v, is_deletion):
    """Add or remove undirected edge (u,v) from edge_index.

    Args:
        edge_index: (2, E) tensor
        u, v: node indices
        is_deletion: if True, remove (u,v) and (v,u); if False, add them
    Returns:
        new edge_index tensor
    """
    if is_deletion:
        mask = ~(
            ((edge_index[0] == u) & (edge_index[1] == v))
            | ((edge_index[0] == v) & (edge_index[1] == u))
        )
        return edge_index[:, mask]
    else:
        device = edge_index.device
        new_edges = torch.tensor([[u, v], [v, u]], device=device).t()
        return torch.cat([edge_index, new_edges], dim=1)


def make_differentiable_adj(edge_index, num_nodes):
    """Create a dense float adjacency matrix with requires_grad=True."""
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj.requires_grad_(True)
    return adj
