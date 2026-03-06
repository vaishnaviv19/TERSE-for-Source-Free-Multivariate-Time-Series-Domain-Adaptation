"""
Spatial GNN Module

Applies graph convolution using the learned adjacency matrix
to capture spatial relationships between sensor nodes.

Input:
    Node features from Temporal CNN
    Shape: (batch_size, num_nodes, feature_dim)

    Adjacency matrix from Graph Learner
    Shape: (batch_size, num_nodes, num_nodes)

Output:
    Updated node features
    Shape: (batch_size, num_nodes, out_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGNN(nn.Module):
    """
    Spatial Graph Neural Network layer that performs
    graph convolution using the learned adjacency matrix.
    """

    def __init__(self, in_features=64, hidden_features=64, out_features=128, dropout=0.3):
        super(SpatialGNN, self).__init__()

        # Graph convolution layers
        self.gcn1 = nn.Linear(in_features, hidden_features)
        self.gcn2 = nn.Linear(hidden_features, out_features)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Forward pass.

        Args:
            x : node features
                shape -> (batch_size, num_nodes, feature_dim)

            adj : adjacency matrix
                shape -> (batch_size, num_nodes, num_nodes)

        Returns:
            Updated node features
        """

        # -------- Graph Convolution Layer 1 --------
        x = torch.matmul(adj, x)  # message passing

        x = self.gcn1(x)

        # reshape for batchnorm
        B, N, feat_dim = x.shape
        x = x.view(B * N, feat_dim)
        x = self.bn1(x)
        x = x.view(B, N,feat_dim)

        x = F.relu(x)
        x = self.dropout(x)

        # -------- Graph Convolution Layer 2 --------
        x = torch.matmul(adj, x)

        x = self.gcn2(x)

        B, N, feat_dim = x.shape
        x = x.view(B * N, feat_dim)
        x = self.bn2(x)
        x = x.view(B, N, feat_dim)

        x = F.relu(x)

        return x


if __name__ == "__main__":

    # Example test
    batch_size = 32
    num_nodes = 6
    feature_dim = 64

    node_features = torch.randn(batch_size, num_nodes, feature_dim)

    adjacency = torch.randn(batch_size, num_nodes, num_nodes)

    model = SpatialGNN()

    output = model(node_features, adjacency)

    print("Node feature shape:", node_features.shape)
    print("Adjacency shape:", adjacency.shape)
    print("Output shape:", output.shape)