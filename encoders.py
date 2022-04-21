import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GINConv, global_add_pool, GATConv
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GINEncoder(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hid_channels, out_channels):
        super(GINEncoder, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(GINConv(Sequential(
                Linear(self.in_channels, self.hid_channels), BatchNorm1d(self.hid_channels), ReLU(),
                           Linear(self.hid_channels, self.out_channels), ReLU())))
        else:
            self.layers.append(GINConv(Sequential(
                Linear(self.in_channels, self.hid_channels), BatchNorm1d(self.hid_channels), ReLU(),
                Linear(self.hid_channels, self.hid_channels), ReLU())))

        if num_layers > 2:
            for i in range(num_layers-2):
                self.layers.append(GINConv(
                Sequential(Linear(self.hid_channels, self.hid_channels), BatchNorm1d(self.hid_channels), ReLU(),
                           Linear(self.hid_channels, self.hid_channels), ReLU())))

        if self.num_layers > 1:
            self.layers.append(GINConv(
                Sequential(Linear(self.hid_channels, self.hid_channels), Linear(self.hid_channels, self.out_channels))))

    def forward(self, x, edge_index):
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)

        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, heads, out_channels, dropout):
        super(GATEncoder, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.heads = heads
        self.conv1 = GATConv(self.in_channels, self.hid_channels, heads=self.heads, dropout=self.dropout)
        self.conv2 = GATConv(self.hid_channels*self.heads, self.out_channels, concat=False, heads=1, dropout=self.dropout)


    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x



