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


def json_to_sparse_matrix(file_dir):
    adj_features_list = []
    for file in tqdm(os.listdir(file_dir)):
        with open(os.path.join(file_dir, file), 'r') as f:
            json_dict = json.loads(json.load(f))
            print(f'Loading {len(json_dict)} items from file')
            for key, _ in json_dict.items():
                sparse_adj = nx.to_scipy_sparse_matrix(json_graph.adjacency_graph(json_dict[key]))
                adj_norm = preprocess_graph(sparse_adj)
                adj_label = sparse_adj + sp.eye(sparse_adj.shape[0])
                adj_label = torch.FloatTensor(adj_label.toarray())
                pos_weight = torch.tensor([float(sparse_adj.shape[0] * sparse_adj.shape[0] - sparse_adj.sum()) / sparse_adj.sum()])
                norm = sparse_adj.shape[0] * sparse_adj.shape[0] / float((sparse_adj.shape[0] * sparse_adj.shape[0] - sparse_adj.sum()) * 2)
                x = torch.ones([sparse_adj.shape[0], 1])
                edge_index, edge_attrs = from_scipy_sparse_matrix(sparse_adj)
                data = Data(x=x, edge_index=edge_index)
                data.adj_norm = adj_norm
                data.adj_label = adj_label
                data.pos_weight = pos_weight
                data.norm = norm
                adj_features_list.append(data)

    return adj_features_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir")
    args = parser.parse_args()
    data_list = json_to_sparse_matrix(args.file_dir)
    adj, features = load_data('cora')
    loader = DataLoader(data_list, batch_size=32)
    for i, batch in enumerate(loader):

        print(i, batch.num_graphs)
