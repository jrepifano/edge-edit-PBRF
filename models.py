import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class VanillaGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Dense layers mirroring convs (for forward_dense)
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels, bias=True))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
        self.lins.append(nn.Linear(hidden_channels, out_channels, bias=True))

        self._sync_dense_from_sparse()

    def _sync_dense_from_sparse(self):
        """Copy weights from GCNConv layers to dense Linear layers."""
        with torch.no_grad():
            for conv, lin in zip(self.convs, self.lins):
                lin.weight.data.copy_(conv.lin.weight.data)
                if conv.bias is not None and lin.bias is not None:
                    lin.bias.data.copy_(conv.bias.data)

    def _sync_sparse_from_dense(self):
        """Copy weights from dense Linear layers to GCNConv layers."""
        for conv, lin in zip(self.convs, self.lins):
            conv.lin.weight.data.copy_(lin.weight.data)
            if conv.bias is not None and lin.bias is not None:
                conv.bias.data.copy_(lin.bias.data)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

    def forward_dense(self, x, adj):
        """Forward pass using dense adjacency matrix for gradient computation.

        adj: (N, N) dense adjacency matrix (without self-loops).
        Uses D^{-1/2} A_hat D^{-1/2} normalization where A_hat = adj + I.
        """
        self._sync_dense_from_sparse()
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj_hat.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt.unsqueeze(1) * adj_hat * deg_inv_sqrt.unsqueeze(0)

        for i, lin in enumerate(self.lins[:-1]):
            # Match GCNConv: linear (no bias) -> propagate -> add bias
            x = x @ lin.weight.t()
            x = norm @ x
            x = x + lin.bias
            x = F.relu(x)
        x = x @ self.lins[-1].weight.t()
        x = norm @ x
        x = x + self.lins[-1].bias
        return x

    def dense_params(self):
        """Return parameters used in forward_dense (the linear layers)."""
        return list(self.lins.parameters())

    def sparse_params(self):
        """Return parameters used in sparse forward (the GCN conv layers)."""
        return list(self.convs.parameters())


def train_model(model, data, lr=0.01, weight_decay=1e-5, epochs=2000, verbose=True):
    """Train a GCN model on graph data using SGD."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Sync dense layers after each step
        model._sync_dense_from_sparse()

        if verbose and (epoch + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                val_pred = logits[data.val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
                train_pred = logits[data.train_mask].argmax(dim=1)
                train_acc = (
                    (train_pred == data.y[data.train_mask]).float().mean().item()
                )
            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )
            model.train()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        val_pred = logits[data.val_mask].argmax(dim=1)
        val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
        test_pred = logits[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
    print(f"Final Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    return model
