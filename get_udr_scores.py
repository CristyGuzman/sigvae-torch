import argparse
import json
import os
import torch
from model_vgae import DeepVGAE
import torch_geometric.transforms as T
from data import json_to_sparse_matrix
from configuration import Configuration, CONSTANTS as C
from disentanglement import compute_udr_sklearn, spearman_correlation_conv, relative_strength_disentanglement
from torch_geometric.loader import DataLoader
import numpy as np

parser = argparse.ArgumentParser()

def parse_cmd():
    #General
    parser.add_argument('--model_dirs', nargs="+", help='list of model directories')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='/home/csolis/data/pyg_datasets')
    args = parser.parse_args()
    return args


def get_model(model_dir, config):
    model = DeepVGAE(config).to(C.DEVICE)
    model_dir = os.path.join(model_dir, 'model.pth')
    loaded_dict = torch.load(model_dir)
    model.load_state_dict(loaded_dict['model_state_dict'])
    return model

def get_representation_functions(model_dir_list, config_list):
    representation_funcs = []
    for model_dir, config in zip(model_dir_list, config_list):
        representation_funcs.append(get_model(model_dir=model_dir, config=config))
    return representation_funcs

def load_embeddings(embedding_dir):
    with open(embedding_dir, 'r') as f:
        emb_dict = json.load(f)
    return emb_dict
def split_embeddings_from_batch(embeddings, batch):
    num_graphs = np.unique(batch).shape[0]
    separate_embeddings = []
    #latent_dim = embeddings.shape[1]
    for i in range(num_graphs):
        mask = np.array(batch) == i
        #mask = np.tile(mask, (latent_dim, 1)).transpose()
        separate_embeddings.append(embeddings[mask,:])
    return separate_embeddings

def aggregate_node_embeddings(graphs_list):
    graph_embeddings = []
    for i in range(len(graphs_list)):
        graph_embeddings.append(np.mean(graphs_list[i], axis=0))
    return graph_embeddings

def get_kl_and_embedding_per_graph(model, data):
    """
    returns num graphs, num latent dims
    """
    z = model.encode(data.x, data.edge_index).detach().numpy()
    batch = data.batch
    mu = model.__mu__.detach().numpy()
    logstd = model.__logstd__.detach().numpy()
    separate_embeddings = split_embeddings_from_batch(z, batch)
    graph_embeddings = aggregate_node_embeddings(separate_embeddings)
    separate_mus = split_embeddings_from_batch(mu, batch)
    separate_logstds = split_embeddings_from_batch(logstd, batch) # list of len num of graphs, each elem (num_nodes, num_latent_dims)
    #torch.mean(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=0).shape
    kl_per_dim = lambda mu, log: 1 + 2 * log - mu ** 2 - np.exp(log) ** 2
    kls_per_graph = []
    for mus, logs in zip(separate_mus, separate_logstds):
        all_kls = kl_per_dim(mus, logs)  # num_nodes, num_latent_dims
        kls_per_graph.append(np.mean(all_kls, axis=0))  # num_latent_dims
    return kls_per_graph, graph_embeddings  # num_graphs,num_latent_dims


#1. get latents from model.encode
#2. get mu and sigma from model
#3. split these in a list of length num_graphs, end up having (num_graphs, (num_nodes, num_latent_dims))
#4. compute kl per dimension (get mean across nodes, end up with (num_graphs, num_latent_dims)
#5. aggregate node embeddings, end up with (num_graphs, num_latent_dims)

# done: get kl and embeddings per graph, calculate udr score:
    # compute correlation matrix

def compute_udr(model_dir_list, data):
    """model_dirs must have config files in them as well
    """
    num_models = len(model_dir_list)
    configs_list = get_configs_list(model_dir_list, data.num_features)
    models = get_representation_functions(model_dir_list, configs_list)
    kl_divergence = []
    representation_points = []
    for j, model in enumerate(models):
        kls_per_graph, graph_embeddings = get_kl_and_embedding_per_graph(model, data)
        total_kls = np.mean(kls_per_graph, axis=0) #num_latent_dims
        kl_divergence.append(kls_per_graph) # num_models, num_latent_dims
        representation_points.append(graph_embeddings)

    kl_divergence = np.array(kl_divergence) # num_models, num_latent_dims
    kls = np.sum(kl_divergence, axis=0) #num_latent_dims
    latent_dim = kl_divergence.shape[1]
    corr_matrix_all = np.zeros((num_models, num_models, latent_dim, latent_dim))
    disentanglement = np.zeros((num_models, num_models, 1))
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue

            corr_matrix = spearman_correlation_conv(representation_points[i], representation_points[j])

            corr_matrix_all[i, j, :, :] = corr_matrix
            disentanglement[i, j] = relative_strength_disentanglement(corr_matrix)
    scores_dict = {}

    scores_dict["raw_correlations"] = corr_matrix_all.tolist()
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()

    model_scores = []
    for i in range(num_models):
        model_scores.append(np.median(np.delete(disentanglement[:, i], i)))

    scores_dict["model_scores"] = model_scores
    return scores_dict


def get_configs_list(model_dir_list, data_input_size):
    configs_list = []
    for model_dir in model_dir_list:
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
            configs_list.append(Configuration(config))
    return configs_list

if __name__ == '__main__':
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True),
    ])
    args = parse_cmd()

    directory = os.path.join(args.data_dir, 'raw')
    file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
    data_list = json_to_sparse_matrix(file_list)
    data_list_transformed = [transform(data) for data in data_list]
    loader = DataLoader(data_list_transformed, batch_size=args.batch_size) #bs train in this case corresponds to
    train_data, val_data, test_data = next(iter(loader))
    model_dir_list = args.model_dirs
    #config_list = [Configuration().to_json(m_dir) for m_dir in model_dir_list]
    # models = get_representation_functions(model_dir_list, config_list)
    #
    # scores = compute_udr_sklearn(train_data,
    #                     models,
    #                     random_state,
    #                     config.bs_train,
    #                     num_data_points=100,#hardcoded 10 graphs per josn file, loading files in /home/csolis/pyg_dataset/raw
    #                     correlation_matrix="spearman",
    #                     filter_low_kl=True,
    #                     include_raw_correlations=True,
    #                     kl_filter_threshold=0.01)
    compute_udr(model_dir_list, train_data)
