### FactorVAE
import  numpy as np
from beta_vae_score import sample_factors
import argparse
import random
from beta_vae_score import sample_from_files
from torch_geometric.loader import DataLoader
from get_udr_scores import get_kl_and_embedding_per_graph, get_configs_list, get_representation_functions
import torch_geometric.transforms as T
from configuration import CONSTANTS as C
import logging
import os
import json

# each data point is an input/output tuple, where the input is the index of
# the dimension from latent representation with the least variance, and the output
# corresponds to the factor of variation that was chosen to generate the data.
#

def compute_factor_vae_score(num_points, seed,):
    pass
def get_representations(model_dir, graphs_factors, batch_size, transform):
    observations1 = [i[0] for i in graphs_factors]
    loader = DataLoader(transform(observations1), batch_size=batch_size, shuffle=True)
    data = next(iter(loader))
    num_features = data.num_features
    configs_list = get_configs_list([model_dir], num_features)
    models = get_representation_functions([model_dir], configs_list)
    return data, models
def get_data_and_models(model_dir, files_dir, batch_size, transform, factors, graphs_per_file):
    """
    Samples generative factors, obtain normalized representations of gernetade data from those factors and returns it
    (as a batch containing all the data) together with the models
    """
    if factors is not None:
        index = random.randint(0, 1)
        # 1. Randomly choose a generative factor
        factors = np.array(sample_factors(batch_size=batch_size))
        # 2. Fix the chosen generative factor and vary all other factors and generate data.
        factors[:, index] = factors[0, index]
        print('start sampling for first list of graphs')
        factors = [list(factor) for factor in factors]
    graphs_factors = sample_from_files(files_dir=files_dir, batch_size=batch_size, factors=factors, graphs_per_file=graphs_per_file)[:batch_size]

    # 3. Obtain representations from the generated data.

    data, models = get_representations(model_dir=model_dir, graphs_factors=graphs_factors, batch_size=batch_size, transform=transform)
    if factors is not None:
        return index, data, models
    else:
        return data, models
def _prune_dims(variances, threshold=0.):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold

def get_training_sample(model_dir, files_dir, batch_size, transform, factors, graphs_per_file=1, global_variances=None, active_dims=None):
    """
    Fixed batch size uses entire dataset FIXME
    """
    if factors is not None:
        index, data, models = get_data_and_models(model_dir=model_dir,
                                                  files_dir=files_dir,
                                                  batch_size=batch_size,
                                                  transform=transform,
                                                  factors=factors,
                                                  graphs_per_file=graphs_per_file)
    else:
        data, models = get_data_and_models(model_dir=model_dir,
                                           files_dir=files_dir,
                                           batch_size=batch_size,
                                           transform=transform,
                                           factors=factors,
                                           graphs_per_file=graphs_per_file)

    model = models[0]
    model.eval()  # we only have one model in list
    _, graph_embeddings = get_kl_and_embedding_per_graph(model, data)  #batch_size, latent dims
    # 4. Normalize the data dividing by its standard deviation over the full data
    graph_embeddings = np.array(graph_embeddings)
    vars = np.var(graph_embeddings, axis=0, ddof=1)
    if factors is None:
        return vars
    else:
        normalized_graph_embeddings = graph_embeddings
        # 5. Take the variance in each dimension and choose argmin
        vars = np.var(normalized_graph_embeddings, axis=0, ddof=1)
        d = np.argmin(vars[active_dims]/global_variances[active_dims])
        return d, index

def generate_training_batch(model_dir, files_dir, batch_size, transform, num_points, latent_dims_vars, num_factors=2, global_variances=None, active_dims=None):
    votes = np.zeros((num_factors, latent_dims_vars.shape[0]),
                     dtype=np.int64)
    for _ in range(num_points):
        argmin, factor_index = get_training_sample(model_dir=model_dir, files_dir=files_dir, batch_size=batch_size,
                                                   factors=1, transform=transform, global_variances=global_variances, active_dims=active_dims)
        votes[factor_index, argmin] += 1
    return votes

def get_global_variances(model_dir, files_dir, sample_size, graphs_per_file, transform):
    global_variances = get_training_sample(model_dir=model_dir,
                                           files_dir=files_dir,
                                           batch_size=sample_size,
                                           transform=transform,
                                           factors=None,
                                           graphs_per_file=graphs_per_file)
    return global_variances


    #get_training_sample(model_dir, files_dir, batch_size, transform)

def compute_factor_vae(training_votes, global_variances, active_dims):
    scores_dict = {}
    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0
        return scores_dict
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])
    logging.info("Evaluate training set accuracy.")
    train_accuracy = np.sum(
        training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    scores_dict['train_accuracy'] = train_accuracy
    scores_dict["num_active_dims"] = len(active_dims)
    return scores_dict


#classifier = np.argmax(training_votes, axis=0)
#other_index = np.arange(training_votes.shape[1])

#logging.info("evaluate training set accuracy.")
#train_accuracy = np.sum(
#    training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
#logging.info("training set accuracy: %.2g", train_accuracy)



def main(args):
    pass

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples for creating one data for computing the score')
    parser.add_argument('--files_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--full_data_sample_size', type=int, help='number of data samples for calculating global variances per latent dimension')
    parser.add_argument('--graphs_per_file', type=int, help='number of graphs to retrieve from json files. Each file containes 10 graphs.')
    parser.add_argument('--num_points', type=int, help='number of points for computing score')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--tag', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cmd()
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),

    ])
    global_variances = get_global_variances(model_dir=args.model_dir,
                                            files_dir=args.files_dir,
                                            sample_size=args.full_data_sample_size,
                                            graphs_per_file=args.graphs_per_file,
                                            transform=transform)
    active_dims = _prune_dims(global_variances)
    training_votes = generate_training_batch(model_dir=args.model_dir,
                                             files_dir=args.files_dir,
                                             batch_size=args.batch_size,
                                             transform=transform,
                                             num_points=args.num_points,
                                             latent_dims_vars=global_variances,
                                             global_variances=global_variances,
                                             active_dims=active_dims)

    scores_dict = compute_factor_vae(training_votes=training_votes, global_variances=global_variances, active_dims=active_dims)
    dirname = args.save_dir + args.tag
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(os.path.join(dirname, f'{args.tag}.json'), 'w') as f:
        json.dump(scores_dict, f)










# 5. Take empirical variance in each dimension of the normalized representations.
