from torch import nn

from models.heads import ClassificationHead
from models.masking import mask_adj_matrices_edges, masking2
from models.spatial import (
    GraphConvolution,
    GraphLearner,
    SpatialGraphEncoder,
    SpatialRewiringNet,
    normalize_batch_adj_both,
)
from models.temporal import TemporalConvEncoder, TemporalRestorationNet


class SpatioTemporalBackbone(nn.Module):
    """TERSE backbone: temporal encoder + graph learner + spatial encoder."""

    def __init__(self, configs):
        super().__init__()
        self.temporal_cnn = TemporalConvEncoder(configs)
        self.graph_learner = GraphLearner(configs)
        self.spatial_gnn = SpatialGraphEncoder(configs)

    def forward(self, x):
        temp_feat = self.temporal_cnn(x)
        adj = self.graph_learner(temp_feat)
        node_feat, flat_feat = self.spatial_gnn(temp_feat, adj)
        return node_feat, flat_feat


# Backward-compatible aliases expected by rest of project.
TemporalCNN = TemporalConvEncoder
Graph_Learner = GraphLearner
Spatial_GNN = SpatialGraphEncoder
GraphRecover_new = SpatialRewiringNet
Temporal_Imputer = TemporalRestorationNet
Classifier = ClassificationHead


