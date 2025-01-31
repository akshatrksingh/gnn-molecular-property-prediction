import torch
import torch.nn.functional as Fun
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(torch.nn.Module):
    """Graph Attention Network class with 3 attention layers and a linear layer,
    following same structure as GCN but with attention mechanism"""

    def __init__(self, dim_h):
        """init method for GAN
        Args:
            dim_h (int): the dimension of hidden layers
        """
        super().__init__()
        # Same as GCN but using GATConv instead of GCNConv
        self.conv1 = GATConv(11, dim_h)
        self.conv2 = GATConv(dim_h, dim_h)
        self.conv3 = GATConv(dim_h, dim_h)
        self.lin = torch.nn.Linear(dim_h, 1)

    def forward(self, data):
        e = data.edge_index
        x = data.x

        # Same structure as GCN
        x = self.conv1(x, e)
        x = x.relu()
        x = self.conv2(x, e)
        x = x.relu()
        x = self.conv3(x, e)
        x = global_mean_pool(x, data.batch)

        x = Fun.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x