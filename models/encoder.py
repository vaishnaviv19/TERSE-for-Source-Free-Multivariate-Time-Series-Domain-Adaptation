"""
Encoder Module

Combines:
1. Temporal CNN
2. Graph Learner
3. Spatial GNN

Input:
    (batch_size, channels, time_steps)

Example:
    (32, 9, 128)

Output:
    Latent embedding
"""

import torch
import torch.nn as nn

from temporal_cnn import TemporalCNN
from graph_learner import GraphLearner
from spatial_gnn import SpatialGNN


class Encoder(nn.Module):

    def __init__(
        self,
        num_nodes=9,              # ← changed from 6 → 9
        temporal_dim=64,
        gnn_hidden_dim=64,
        gnn_output_dim=128
    ):
        super().__init__()

        self.num_nodes = num_nodes

        # Temporal feature extractor
        self.temporal_cnn = TemporalCNN(
            in_channels=num_nodes
        )

        # Graph learner
        self.graph_learner = GraphLearner(
            num_nodes=num_nodes,
            feature_dim=temporal_dim
        )

        # Spatial GNN
        self.spatial_gnn = SpatialGNN(
            in_features=temporal_dim,
            hidden_features=gnn_hidden_dim,
            out_features=gnn_output_dim
        )

        # Pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        """
        x shape:
        (batch_size, 9, 128)
        """

        batch_size = x.size(0)

        # -------- Temporal CNN --------
        temporal_features = self.temporal_cnn(x)

        # reshape to node features
        node_features = temporal_features.view(
            batch_size,
            self.num_nodes,
            -1
        )

        # -------- Graph Learning --------
        adjacency = self.graph_learner(node_features)

        # -------- Spatial GNN --------
        spatial_features = self.spatial_gnn(node_features, adjacency)

        # (batch, nodes, features) → (batch, features, nodes)
        spatial_features = spatial_features.transpose(1, 2)

        # -------- Global Pooling --------
        embedding = self.global_pool(spatial_features)

        embedding = embedding.squeeze(-1)

        return embedding


# -------------------------------------------------------
# Test Run
# -------------------------------------------------------

if __name__ == "__main__":

    batch_size = 32
    channels = 9
    time_steps = 128

    dummy_input = torch.randn(batch_size, channels, time_steps)

    model = Encoder()

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Embedding shape:", output.shape)