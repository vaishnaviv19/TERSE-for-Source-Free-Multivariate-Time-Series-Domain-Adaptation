from torch import nn


class ClassificationHead(nn.Module):
    """Linear classifier on flattened spatial features."""

    def __init__(self, configs):
        super().__init__()
        output_dim = configs.gnn_output_dim
        self.logits = nn.Linear(output_dim * configs.input_channels, configs.num_classes)

    def forward(self, x):
        return self.logits(x)
