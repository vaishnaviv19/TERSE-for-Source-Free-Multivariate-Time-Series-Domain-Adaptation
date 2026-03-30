import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def normalize_batch_adj_both(batch_adj_matrices: torch.Tensor) -> torch.Tensor:
    """Batch normalization for adjacency matrices: D^-1/2 * A * D^-1/2."""
    degree = batch_adj_matrices.sum(dim=2)
    degree_sqrt_inv = torch.where(degree != 0, torch.pow(degree, -0.5), torch.zeros_like(degree))
    d_inv_sqrt = torch.diag_embed(degree_sqrt_inv)
    return torch.matmul(torch.matmul(d_inv_sqrt, batch_adj_matrices), d_inv_sqrt)


class GraphLearner(nn.Module):
    """Builds channel correlation graph via normalized inner product."""

    def __init__(self, _configs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        x_t = x_norm.transpose(1, 2)
        adj = torch.bmm(x_norm, x_t)
        return F.relu(adj)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("mik,kj->mij", inputs, self.weight)
        output = torch.einsum("mki,mij->mkj", adj, support)
        if self.bias is not None:
            output += self.bias
        return output


class SpatialGraphEncoder(nn.Module):
    """Single-layer spatial GNN encoder."""

    def __init__(self, configs):
        super().__init__()
        self.conv = GraphConvolution(configs.gnn_input_dim, configs.gnn_output_dim)
        self.drop = nn.Dropout(configs.dropout_spatial_gnn)
        self.bn = nn.BatchNorm1d(configs.gnn_output_dim)
        self.act = nn.PReLU()
        self.adj_norm = configs.adj_norm

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = self.drop(x)
        adj_in = normalize_batch_adj_both(adj) if self.adj_norm else adj
        h = self.conv(h, adj_in)
        h = torch.transpose(self.bn(h.transpose(1, 2)), 1, 2)
        h = self.act(h)
        h_flat = h.contiguous().view(x.shape[0], -1)
        return h, h_flat


class SpatialRewiringNet(nn.Module):
    """Graph rewiring head used for spatial restoration objective."""

    def __init__(self, configs):
        super().__init__()
        self.conv = GraphConvolution(configs.gnn_output_dim, configs.gnn_output_dim)
        self.drop = nn.Dropout(configs.dropout_graph_recover)
        self.bn = nn.BatchNorm1d(configs.gnn_output_dim)
        self.act = nn.PReLU()
        self.adj_norm = configs.adj_norm

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.drop(x)
        adj_in = normalize_batch_adj_both(adj) if self.adj_norm else adj
        h = self.conv(h, adj_in)
        h = torch.transpose(self.bn(h.transpose(1, 2)), 1, 2)
        h = self.act(h)

        h = F.normalize(h, p=2.0, dim=-1)
        h_t = h.transpose(1, 2)
        new_adj = torch.bmm(h, h_t)
        return F.relu(new_adj)
