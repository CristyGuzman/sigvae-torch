import os
from tqdm import tqdm
import time
import sys

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

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True),
    ])
    # dataset = Planetoid("datasets", args.dataset, transform=transform)
    # data = dataset[0]
    # dataset
    print('Creating training dataset')
    #dataset = MyOwnDataset(root=config.train_data_dir, directory=config.train_data_dir, transform=transform)
    directory = os.path.join(config.train_data_dir, 'raw')
    file_list = [os.path.abspath(f) for f in os.listdir(directory) if os.path.isfile(f)]
    data_list = json_to_sparse_matrix(file_list)
    data_list_transformed = [transform(data) for data in data_list]
    loader = DataLoader(data_list_transformed, batch_size=config.bs_train)

    if config.input_size is None:
        config.input_size = dataset.num_features

    model = DeepVGAE(config).to(C.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.lr)

    experiment_id = int(time.time())
    experiment_name = model.model_name()
    model_dir = create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)

    config.to_json(os.path.join(model_dir, 'config.json'))
    # Save the command line that was used.
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(cmd)

    writer = SummaryWriter(os.path.join(model_dir, 'logs'))

    # train_data, valid_data, test_data = data
    global_step = 0
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
            global_step += 1

if __name__ == '__main__':
    main(Configuration.parse_cmd())
