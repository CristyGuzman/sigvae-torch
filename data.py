import os

from torch_geometric.loader import DataLoader
import networkx as nx
from networkx.readwrite import json_graph
import json
from tqdm import tqdm
import argparse
import torch
from utils import load_data


def json_to_sparse_matrix(file_dir):
    adj_features_list = []
    for file in os.listdir(file_dir):
        with open(os.path.join(file_dir, file), 'r') as f:
            json_dict = json.loads(json.load(f))
            print(f'Loading {len(json_dict)} items from file')
            for key, _ in tqdm(json_dict.items()):
                adj = nx.to_scipy_sparse_matrix(json_graph.adjacency_graph(json_dict[key]))
                features = torch.ones([1, adj.shape[0], 1])
                adj_features_list.append((adj, features))
        return adj_features_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir")
    args = parser.parse_args()
    data_list = json_to_sparse_matrix(args.file_dir)
    adj, features = load_data('cora')
    loader = DataLoader(data_list, batch_size=32)
