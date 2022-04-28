import argparse
import os
import torch
from model_vgae import DeepVGAE
import torch_geometric.transforms as T
from data import json_to_sparse_matrix
from configuration import Configuration, CONSTANTS as C
from disentanglement import compute_udr_sklearn
from torch_geometric.loader import DataLoader
import numpy as np

parser = argparse.ArgumentParser()

def parse_cmd():
    #General
    parser.add_argument('--model_dir1', type=str, help='directory path where first model and config are saved.')
    parser.add_argument('--model_dir2', type=str, help='directory path where second model and config are saved.')
    parser.add_argument('--model_dir3', type=str, help='directory path where third model and config are saved.')
    parser.add_argument('--model_dir4', type=str, help='directory path where fourth model and config are saved.')
    args = parser.parse_args()
    return args


def get_model(model_dir, config):
    model = DeepVGAE(config).to(C.DEVICE)
    model.load_state_dict(torch.load(model_dir))
    return model

def get_representation_functions(model_dir_list, config_list):
    representation_funcs = []
    for model_dir, config in zip(model_dir_list, config_list):
        representation_funcs.append(get_model(model_dir=model_dir, config=config))
    return representation_funcs


if __name__ == '__main__':
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True),
    ])
    args = parse_cmd()
    config = Configuration().to_json(args.model_dir)

    directory = os.path.join(config.train_data_dir, 'raw')
    file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
    data_list = json_to_sparse_matrix(file_list)
    data_list_transformed = [transform(data) for data in data_list]
    loader = DataLoader(data_list_transformed, batch_size=config.bs_train) #bs train in this case corresponds to
    train_data, val_data, test_data = next(iter(loader))
    random_state = np.random.RandomState(0)
    model_dir_list = [args.model_dir_1, args.model_dir_2, args.model_dir_3, args.model_dir_4]
    config_list = [Configuration().to_json(m_dir) for m_dir in model_dir_list]
    models = get_representation_functions(model_dir_list, config_list)

    scores = compute_udr_sklearn(train_data,
                        models,
                        random_state,
                        config.bs_train,
                        num_data_points=100,#hardcoded 10 graphs per josn file, loading files in /home/csolis/pyg_dataset/raw
                        correlation_matrix="spearman",
                        filter_low_kl=True,
                        include_raw_correlations=True,
                        kl_filter_threshold=0.01)


