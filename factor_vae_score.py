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

# each data point is an input/output tuple, where the input is the index of
# the dimension from latent representation with the least variance, and the output
# corresponds to the factor of variation that was chosen to generate the data.
#

def compute_factor_vae_score(num_points, seed,):
    pass

def get_training_sample(model_dir, files_dir, batch_size, transform):
    index = random.randint(0, 1)
    # 1. Randomly choose a generative factor
    factors = np.array(sample_factors(batch_size=batch_size))
    # 2. Fix the chosen generative factor and vary all other factors and generate data.
    factors[:, index] = factors[0, index]
    print('start sampling for first list of graphs')
    factors = [list(factor) for factor in factors]
    graphs_factors = sample_from_files(files_dir=files_dir, batch_size=batch_size, factors=factors)[:batch_size]
    # 3. Obtain representations from the generated data.
    observations1 = [i[0] for i in graphs_factors]
    loader = DataLoader(transform(observations1), batch_size=batch_size, shuffle=True)
    data = next(iter(loader))
    configs_list = get_configs_list([model_dir], data.num_features)
    models = get_representation_functions([model_dir], configs_list)
    model = models[0]
    model.eval()  # we only have one model in list
    _, graph_embeddings = get_kl_and_embedding_per_graph(model, data)
    # 4. Normalize the data dividing by its standard deviation over the full data


def main(args):
    pass

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples for creating one data for computing the score')
    parser.add_argument('--files_dir', type=str)
    parser.add_argument('--model_dir', type=str)


if __name__ == "__main__":
    args = parse_cmd()
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),

    ])
    get_training_sample(args.model_dir, args.files_dir, args.batch_size, transform)









# 5. Take empirical variance in each dimension of the normalized representations.
