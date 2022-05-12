import sklearn
import os
import numpy as np
import re
from tqdm import tqdm
import random
import json
from data import json_to_sparse_matrix, create_data_object
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from get_udr_scores import get_kl_and_embedding_per_graph, get_model, get_configs_list, get_representation_functions
from configuration import CONSTANTS as C
from sklearn import linear_model
import logging
import argparse

def sample_factors(batch_size, fixed_param=None):
    """
    function taht returns a list of tuples, where first element is a number between 2 and 10 corresponding to the k param,
    second element in tuple is p. List of length batch_size.
    """
    if fixed_param is None:
        return [(random.randint(2, 10), random.uniform(0, 1)) for _ in range(batch_size)]
    elif fixed_param == 0:
        return [random.uniform(0, 1) for _ in range(batch_size)]
    else:
        if fixed_param != 1:
            raise ValueError("chosen param index must be either 0 or 1")
        return [random.randint(2, 10) for _ in range(batch_size)]




def sample_from_files(files_dir, batch_size, factors):
    #1. sample batch_size graphs. since files contain 10 graphs need to randomly take 1 per file, so do batch_size json loads
    # append batch_size graphs and convert list to batch
    """
    samples batch_size generative factors and generates one graph per factor
    ONLY FOR WS GRAPHS
    0. Choose generative factor g (can be either k or p)
    1. Sample two pairs of (k,p) factors
    2. for each pair, sample batch_size graphs from filtered files that contain such generative factors
    """
    ##
    #factors = sample_factors(batch_size=batch_size) #list with batch_size elements
    filenames = sorted(os.listdir(files_dir))
    #print(f'chosen params: k={[i[0] for i in factors]}\n p={[i[1] for i in factors]}')
    filtered_files = []
    for factor in factors: #loops batch_size numebr of times
        for file in tqdm(filenames):
            file_no_ext = re.sub('_.json', '', file)
            list_params = file_no_ext.split('__')
            p_file = round(float(re.sub('_', '.', list_params[-1])), 1)
            k = int(list_params[-2])
            if (p_file == round(factor[1])) & (k == factor[0]):
                print(file, (p_file, factor[1]), (k, factor[0]))
                filtered_files.append((file, factor)) #list of tuples, first elem a file, second element another tuple with the params
    #filtered_files = filtered_files[:batch_size] #to keep batch_size fixed
    graph_factor_list = []
    for file, factor in tqdm(filtered_files):
        with open(os.path.join(files_dir, file), 'r') as f:
            json_dict = json.loads(json.load(f))
            keys = list(json_dict.keys())
            key = random.randrange(len(keys)) #choose one graph from file
            json_dict_val = json_dict[str(key)]
            data = create_data_object(json_dict_val)
            graph_factor_list.append((data, factor))

    return graph_factor_list

def create_pairs(batch_size, index):

    factors1 = sample_factors(batch_size=batch_size) #list of batch_size pairs of factors
    fixed_factors2 = [i[index] for i in factors1]

    variable_factors2 = sample_factors(batch_size=batch_size, fixed_param=index)
    if index == 0:
        factors2 = [(k, p) for k, p in zip(fixed_factors2, variable_factors2)]
    else:
        factors2 = [(k, p) for k, p in zip(variable_factors2, fixed_factors2)]
    # all elements from the list will have the same value for the fixed generative factor
    return (factors1, factors2)

def get_training_sample(model_dir, batch_size, files_dir, transform):
    """
    for each batch, generate a
    create two lists of length batch_size
     3. append graphs to list and convert to batch object
    Returns: tuple with g value (either 0 or 1) and Batch object with batch_size graphs
    """
    index = random.randint(0, 1) #2 generative factors for ws graphs
    factors1, factors2 = create_pairs(batch_size=batch_size, index=index)
    print(factors1, factors2)
    print('start sampling for first list of graphs')
    graph_factor_list_1 = sample_from_files(files_dir=files_dir, batch_size=batch_size, factors=factors1)[:batch_size]
    print('start sampling for second list')
    graph_factor_list_2 = sample_from_files(files_dir=files_dir, batch_size=batch_size, factors=factors2)[:batch_size]
    factors1 = [i[1][index] for i in graph_factor_list_1]
    factors2 = [i[1][index] for i in graph_factor_list_2]
    print(factors1, factors2)
    if factors1 != factors2:
        raise ValueError("Generative factors must be the same")
    observations1 = [i[0] for i in graph_factor_list_1]
    observations2 = [i[0] for i in graph_factor_list_2]
    loader1 = DataLoader(transform(observations1), batch_size=batch_size, shuffle=True)
    loader2 = DataLoader(transform(observations2), batch_size=batch_size, shuffle=True)
    data1 = next(iter(loader1))
    data2 = next(iter(loader2))
    configs_list = get_configs_list([model_dir], data1.num_features)
    models = get_representation_functions([model_dir], configs_list)
    model = models[0]
    model.eval() #we only have one model in list
    _, graph_embeddings1 = get_kl_and_embedding_per_graph(model, data1)
    _, graph_embeddings2 = get_kl_and_embedding_per_graph(model, data2)
    graph_embeddings1 = np.array(graph_embeddings1)
    graph_embeddings2 = np.array(graph_embeddings2)
    feature_vector = np.mean(np.abs(graph_embeddings1 - graph_embeddings2), axis=0)
    return index, feature_vector

def get_training_batch(num_points, batch_size, files_dir, model_dir, transform):
    points = None  # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64)
    for i in range(num_points):
        labels[i], feature_vector = get_training_sample(model_dir=model_dir, batch_size=batch_size, files_dir=files_dir, transform=transform)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels

def compute_beta_vae_score(files_dir, model_dir, batch_size, num_points, transform, random_state):

    train_points, train_labels = get_training_batch(num_points=num_points,
                                                    batch_size=batch_size,
                                                    files_dir=files_dir,
                                                    model_dir=model_dir,
                                                    transform=transform)

    model = linear_model.LogisticRegression(random_state=random_state)
    model.fit(train_points, train_labels)

    logging.info("Evaluate training set accuracy.")
    train_accuracy = np.mean(model.predict(train_points) == train_labels)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    test_points, test_labels = get_training_batch(num_points=num_points,
                                                  batch_size=batch_size,
                                                  files_dir=files_dir,
                                                  model_dir=model_dir,
                                                  transform=transform)
    logging.info("Evaluate test set accuracy.")
    test_accuracy = np.mean(model.predict(test_points) == test_labels)
    logging.info("Test set accuracy: %.2g", test_accuracy)
    scores_dict = {}
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = test_accuracy
    return scores_dict



def parse_cmd():
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument('--model_dir', nargs="+", help='list of model directories')
    parser.add_argument('--files_dir', type=str)
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='/home/csolis/data/pyg_datasets')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_cmd()
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),

    ])
    model_dir_list = args.model_dir
    num_models = len(args.model_dir)
    for i, model_dir in enumerate(model_dir_list):
        print(f'Model {i}')
        random_state = np.random.RandomState(args.seed)
        scores_dict = compute_beta_vae_score(files_dir=args.files_dir,
                               model_dir=model_dir,
                               batch_size=args.batch_size,
                               num_points=args.num_points,
                               transform=transform,
                               random_state=random_state)

        dirname = args.save_dir + args.tag
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(os.path.join(dirname, f'beta_vae_{i}.json'), 'w') as f:
            json.dump(scores_dict, f)
