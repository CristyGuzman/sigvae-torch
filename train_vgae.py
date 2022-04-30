import os
from tqdm import tqdm
import time
import sys
import json
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from model_vgae import DeepVGAE
from vgae_example import create_model_dir
from configuration import CONSTANTS as C
from configuration import Configuration

from data import MyOwnDataset, json_to_sparse_matrix
from torch_geometric.loader import DataLoader

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




os.makedirs("datasets", exist_ok=True)
def main(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
    ])
    random_link_split = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True)
    # dataset = Planetoid("datasets", args.dataset, transform=transform)
    # data = dataset[0]
    # dataset
    print('Creating training dataset')
    #dataset = MyOwnDataset(root=config.train_data_dir, directory=config.train_data_dir, transform=transform)
    directory = os.path.join(config.train_data_dir, 'raw')
    file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
    data_list = json_to_sparse_matrix(file_list)
    data_list_transformed = [transform(data) for data in data_list]
    data_list_split = [random_link_split(data) for data in data_list_transformed]
    loader = DataLoader(data_list_split, batch_size=config.bs_train)
    test_loader = DataLoader(data_list_transformed, batch_size=config.bs_eval)
    if config.input_size is None:
        config.input_size = data_list[0].num_features

    model = DeepVGAE(config).to(C.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.lr)

    experiment_id = int(time.time())
    experiment_name = model.model_name()
    model_dir = create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)
    emb_directory = os.path.join(model_dir, 'embeddings')
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    print('Saving checkpoints to {}'.format(checkpoint_file))

    config.to_json(os.path.join(model_dir, 'config.json'))
    # Save the command line that was used.
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(cmd)

    writer = SummaryWriter(os.path.join(model_dir, 'logs'))

    # train_data, valid_data, test_data = data
    global_step = 0
    best_loss = float('inf')
    for epoch in range(config.n_epochs):
        print(f'Epoch {epoch}')
        for i, abatch in tqdm(enumerate(loader)):
            train_data, val_data, test_data = abatch
            all_edge_index = train_data.edge_index
            model.train()
            optimizer.zero_grad()
            pos_loss, neg_loss, kl_loss, loss, losses = model.loss(train_data.x, train_data.pos_edge_label_index, all_edge_index)
            for k in losses:
                prefix = '{}/{}'.format(k, 'train')
                writer.add_scalar(prefix, losses[k], global_step)
            loss.backward()
            optimizer.step()
            if global_step % config.eval_every == 0:
                model.eval()
                roc_auc, ap = model.single_test(train_data.x,
                                                train_data.pos_edge_label_index,
                                                test_data.pos_edge_label_index,
                                                test_data.neg_edge_label_index)
                writer.add_scalar('ROC', roc_auc, global_step)
                writer.add_scalar('ap', ap, global_step)
                print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
                pos_loss, neg_loss, kl_loss, loss, losses = model.loss(train_data.x, train_data.pos_edge_label_index,
                                                                       all_edge_index)
                if float(loss) < best_loss:
                   best_loss = float(loss)
                   torch.save({
                       'iteration': i,
                       'epoch': epoch,
                       'global_step': global_step,
                       'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'total_loss': float(loss),
                       'kl_loss': float(kl_loss),
                       'recon_loss': float(pos_loss) + float(neg_loss),
                   }, checkpoint_file)

            global_step += 1
    print("Finsihed training. Get embeddings for all graphs and save them...")
    model.eval()
    for i, abatch in tqdm(enumerate(test_loader)):
        latents = model.encode(abatch.x, abatch.edge_index)
        latents_dict = {'embedding': latents.detach().numpy(),
                        'batch': abatch.batch}
        print(latents_dict)
        if not os.path.exists(emb_directory):
            os.makedirs(emb_directory)
        with open(os.path.join(emb_directory, f'emb_{i}.pkl'), 'wb') as fp:
            pickle.dump(latents_dict, fp)

if __name__ == '__main__':
    main(Configuration.parse_cmd())
