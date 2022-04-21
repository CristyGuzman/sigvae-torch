import argparse
import os.path as osp
from data import MyOwnDataset
import torch
from tqdm import tqdm
from utils import load_data, preprocess_graph
from data import json_to_sparse_matrix
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GINConv, global_add_pool
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


class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'gcn':
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        elif self.encoder_type == 'gin':
            self.conv1 = GINEncoder(2, in_channels, 2 * out_channels, 2 * out_channels)
            self.conv_mu = GINEncoder(1, 2 * out_channels, 2 * out_channels, out_channels)
            self.conv_logstd = GINEncoder(1, 2 * out_channels, out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


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





def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', default='gcn')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--validation_steps', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = MyOwnDataset(root='/home/csolis/data/pyg_datasets/', transform=transform)
    dataset[0]
    loader = DataLoader(dataset, batch_size=3)

    in_channels, out_channels = dataset.num_features, 16

    model = VGAE(VariationalEncoder(in_channels, out_channels, encoder_type=args.encoder_type))


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        for i, data in tqdm(enumerate(loader)):
            train_data, val_data, test_data = data
            loss = train()
            print(f'Loss: {loss:.4f}\n')
            print(f'Epoch: {epoch:03d}')
            if i % args.validation_steps == 0:
                auc, ap = test(test_data)
                print(f'Iteration: {i:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')




