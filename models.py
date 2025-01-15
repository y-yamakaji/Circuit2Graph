"""Module providing GNN models """
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_scatter import scatter_max

class SAGE(nn.Module):
    def __init__(self, in_num, out_num):
        super().__init__()

        self.conv1 = SAGEConv(in_channels=in_num, out_channels=16, aggr='mean')
        self.conv2 = SAGEConv(16, 64, aggr='mean')
        self.conv3 = SAGEConv(64, 512, aggr='mean')

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, out_features=out_num)

        self.batch1 = BatchNorm(16)
        self.batch2 = BatchNorm(64)
        self.batch3 = BatchNorm(512)

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, data):
        '''
        GNN process
        '''
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.batch1(x).relu()
        x = self.conv2(x, edge_index)
        x = self.batch2(x).relu()
        x = self.dropout1(x)
        x = self.conv3(x, edge_index)
        x = self.batch3(x).relu()
        x, _ = scatter_max(x, batch, dim=0)
        x = self.linear1(x).relu()
        x = self.dropout2(x)
        x = self.linear2(x)
        return x
