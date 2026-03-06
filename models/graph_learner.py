"""
Graph Learner Module

Learns the adjacency matrix (spatial relationships) between
sensor channels using feature similarity.

Input:
    Node features from Temporal CNN

    Shape:
        (batch_size, num_nodes, feature_dim)

Example:
    (32, 6, 64)

Output:
    Learned adjacency matrix

    Shape:
        (batch_size, num_nodes, num_nodes)

This module models spatial correlations between sensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(nn.Module):
    """
    Graph Learner that constructs an adjacency matrix
    based on similarity between node features.
    """

    def __init__(self, num_nodes=6, feature_dim=64, normalize=True):
        super(GraphLearner, self).__init__()

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.normalize = normalize

        # Optional projection layer (can improve learning)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Node features
               shape -> (batch_size, num_nodes, feature_dim)

        Returns:
            adjacency matrix
               shape -> (batch_size, num_nodes, num_nodes)
        """

        # Optional feature projection
        x_proj = self.proj(x)

        # Compute similarity using inner product
        adjacency = torch.matmul(x_proj, x_proj.transpose(1, 2))

        # Normalize adjacency if required
        if self.normalize:
            adjacency = F.softmax(adjacency, dim=-1)

        return adjacency


if __name__ == "__main__":

    # Example test
    batch_size = 32
    num_nodes = 6
    feature_dim = 64

    dummy_input = torch.randn(batch_size, num_nodes, feature_dim)

    model = GraphLearner(num_nodes=num_nodes, feature_dim=feature_dim)

    adjacency = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Adjacency shape:", adjacency.shape)