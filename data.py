import os
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import networkx as nx
from networkx.readwrite import json_graph
import json
from tqdm import tqdm
import argparse
import scipy.sparse as sp
import torch
from utils import load_data, preprocess_graph
from torch_geometric.utils import train_test_split_edges
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if (key == 'pos_weight') or (key == 'norm') or (key == 'adj_norm') or (key == 'adj_label'):
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def stores_as(self, data: 'BaseData'):
        return self
def create_data_object(json_dict_val):

    sparse_adj = nx.to_scipy_sparse_matrix(json_graph.adjacency_graph(json_dict_val))
    adj_norm = preprocess_graph(sparse_adj)
    adj_norm = torch.unsqueeze(adj_norm, dim=0)
    adj_label = sparse_adj + sp.eye(sparse_adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    adj_label = torch.unsqueeze(adj_label, dim=0)
    pos_weight = torch.tensor([float(sparse_adj.shape[0] * sparse_adj.shape[0] - sparse_adj.sum()) / sparse_adj.sum()])
    pos_weight = torch.unsqueeze(pos_weight, dim=0)
    norm = sparse_adj.shape[0] * sparse_adj.shape[0] / float(
        (sparse_adj.shape[0] * sparse_adj.shape[0] - sparse_adj.sum()) * 2)
    norm = torch.unsqueeze(torch.tensor(norm), dim=0)
    x = torch.transpose(torch.eye(1000)[:sparse_adj.shape[0]], 0, 1)
    edge_index, edge_attrs = from_scipy_sparse_matrix(sparse_adj)
    data = MyData(x=x.T, edge_index=edge_index)
    # data.adj_norm = adj_norm
    # data.adj_label = adj_label
    # data.pos_weight = pos_weight
    # data.norm = norm
    return data

def json_to_sparse_matrix(file_dir):
    adj_features_list = []
    #for file in tqdm(os.listdir(file_dir)):
    for file in tqdm(file_dir):
        with open(file, 'r') as f:
            json_dict = json.loads(json.load(f))
            for key, _ in json_dict.items():
                sparse_adj = nx.to_scipy_sparse_matrix(json_graph.adjacency_graph(json_dict[key]))
                adj_norm = preprocess_graph(sparse_adj)
                adj_norm = torch.unsqueeze(adj_norm, dim=0)
                adj_label = sparse_adj + sp.eye(sparse_adj.shape[0])
                adj_label = torch.FloatTensor(adj_label.toarray())
                adj_label = torch.unsqueeze(adj_label, dim=0)
                pos_weight = torch.tensor([float(sparse_adj.shape[0] * sparse_adj.shape[0] - sparse_adj.sum()) / sparse_adj.sum()])
                pos_weight = torch.unsqueeze(pos_weight, dim=0)
                norm = sparse_adj.shape[0] * sparse_adj.shape[0] / float((sparse_adj.shape[0] * sparse_adj.shape[0] - sparse_adj.sum()) * 2)
                norm = torch.unsqueeze(torch.tensor(norm), dim=0)
                x = torch.transpose(torch.eye(1000)[:sparse_adj.shape[0]], 0, 1)
                edge_index, edge_attrs = from_scipy_sparse_matrix(sparse_adj)
                data = MyData(x=x.T, edge_index=edge_index)
                #data.adj_norm = adj_norm
                #data.adj_label = adj_label
                #data.pos_weight = pos_weight
                #data.norm = norm
                adj_features_list.append(data)

    return adj_features_list


import torch
from torch_geometric.data import InMemoryDataset, download_url


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, directory, transform=None, pre_transform=None, pre_filter=None):
        self.raw_file_names = os.listdir(os.path.join(directory, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.dir = directory
        #self.file_dir = file_dir

    @property
    def raw_file_names(self):
        return self.__raw_file_names
        #if type == 'train':
        #    return os.listdir('/home/csolis/data/pyg_datasets/train/raw')
        #elif type == 'valid':
        #    return os.listdir('/home/csolis/data/pyg_datasets/valid/raw')

    @raw_file_names.setter
    def raw_file_names(self, value):
        self.__raw_file_names = value
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_paths))]


    #def download(self):
        # Download to `self.raw_dir`.
    #    download_url(url, self.raw_dir)
    #    ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = json_to_sparse_matrix(self.raw_paths)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #print(self.processed_paths)
        #for i, data in enumerate(data_list):
        #    torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
        torch.save((data, slices), self.processed_paths[0])





if __name__ == '__main__':
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    #dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    #loader = DataLoader(dataset, batch_size=32, shuffle=True)



    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir")
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
    #dataset = MyOwnDataset(root='/home/csolis/data/pyg_datasets/')
    data_list = json_to_sparse_matrix(args.file_dir)

    loader = DataLoader(data_list, batch_size=4)
    adj, features = load_data('cora')
    for i, batch in enumerate(loader):
        batch = train_test_split_edges(batch)
        print(i, batch.num_graphs)
