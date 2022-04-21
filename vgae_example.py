import argparse
import os
import pickle
import shutil
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
from encoders import GINEncoder, GCNEncoder, GATEncoder



class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, encoder_type, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'gcn':
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        elif self.encoder_type == 'gin':
            if 'num_layers' not in kwargs:
                raise ValueError('Number of layers were not provided')
            else:
                self.num_layers = kwargs['num_layers']
            self.conv1 = GINEncoder(self.num_layers, in_channels, 2 * out_channels, 2 * out_channels)
            self.conv_mu = GINEncoder(1, 2 * out_channels, 2 * out_channels, out_channels)
            self.conv_logstd = GINEncoder(1, 2 * out_channels, out_channels, out_channels)
        elif self.encoder_type == 'gat':
            if 'dropout' not in kwargs:
                raise ValueError('Dropout not provided')
            else:
                self.dropout = kwargs['dropout']
            self.conv1 = GATEncoder(in_channels=in_channels, hid_channels=2*out_channels, heads=5, out_channels=2*out_channels, dropout=self.dropout)
            self.conv_mu = GATEncoder(in_channels=2*out_channels, hid_channels=2*out_channels, heads=5,
                                   out_channels=out_channels, dropout=self.dropout)
            self.conv_logstd = GATEncoder(in_channels=2 * out_channels, hid_channels=2 * out_channels, heads=5,
                                      out_channels=out_channels, dropout=self.dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



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
    parser.add_argument('--dir_train_data', default='/home/csolis/data/pyg_datasets/train')
    parser.add_argument('--dir_test_data', default='/home/csolis/data/pyg_datasets/test')
    parser.add_argument('--out_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--encoder_type', default='gcn')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--validation_steps', type=int, default=10)
    parser.add_argument('--save_embeddings_dir', default='/home/csolis/data/embeddings')
    args = parser.parse_args()
    print(f'Arguments:\n {args}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = MyOwnDataset(root=args.dir_train_data, directory=args.dir_train_data, transform=transform)
    dataset[0]
    loader = DataLoader(dataset, batch_size=args.batch_size)

    in_channels, out_channels = dataset.num_features, args.out_channels
    kwargs = {'dropout': args.dropout, 'num_layers': args.num_layers}
    model = VGAE(VariationalEncoder(in_channels, out_channels, encoder_type=args.encoder_type, **kwargs))


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    losses = []
    aucs = []
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch: {epoch:03d}')
        for i, data in tqdm(enumerate(loader)):
            train_data, val_data, test_data = data
            loss = train()
            losses.append()
            #print(f'Loss: {loss:.4f}')
            if i % args.validation_steps == 0:
                print(loss)
                auc, ap = test(test_data)
                aucs.append(auc)
                #print(f'Iteration: {i:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    print('Saving losses to dir')
    with open('/home/csolis/losses', 'wb') as f:
        pickle.dump(losses)
    with open('/home/csolis/auc', 'wb') as f:
        pickle.dump(aucs)

    # get final embeddings
    transform_test = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])
    dataset_test = MyOwnDataset(root=args.dir_test_data, transform=transform_test)
    loader = DataLoader(dataset_test, batch_size=args.batch_size)
    model.eval()
    if os.path.exists(args.save_embeddings_dir) and os.path.isdir(args.save_embeddings_dir):
        shutil.rmtree(args.save_embeddings_dir)
        os.mkdir(args.save_embedding_dir)
    print('Finished training.')
    for i, data in enumerate(loader):
        z = model.encode(data.x, data.edge_index)
        torch.save(z, os.path.join(args.save_embeddings_dir, f'{args.encoder_type}', f'emb_{i:03d}.pt'))





