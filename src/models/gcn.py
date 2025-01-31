import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    """Graph Convolutional Network class with 3 convolutional layers and a linear layer"""

    def __init__(self, dim_h):
        """Initialize GCN model

        Args:
            input_dim (int): Number of input features
            dim_h (int): Dimension of hidden layers
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(11, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = nn.Linear(dim_h, 1)

    def forward(self, data):
        """Forward pass of GCN model"""
        edge_index = data.edge_index
        x = data.x

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Pooling to get graph-level representation
        x = global_mean_pool(x, data.batch)

        # Apply dropout
        x = F.dropout(x, p=0.5, training=self.training)

        return self.lin(x)