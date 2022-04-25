import argparse
import os
import pickle
import shutil
from data import MyOwnDataset
import torch
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
#from torch_geometric.nn.models.autoencoder import VGAE
from encoders import VariationalEncoder, VGAE2
import yaml
from torch.utils.tensorboard import SummaryWriter
from configuration import Constants as C
from configuration import Configuration
import time
from torch import nn
import sys

#CONFIG_PATH = "./"
def load_config(config_path):
    #with open(os.path.join(CONFIG_PATH, config_name)) as file:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config





def train(model):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    recon_loss = model.recon_loss(z, train_data.pos_edge_label_index)
    kl_loss = model.kl_loss
    loss = recon_loss + (1 / train_data.num_nodes) * kl_loss
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
    optimizer.step()
    return {'total_loss': float(loss), 'recon_loss': float(recon_loss), 'kl_loss': float(kl_loss)}

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

def create_model_dir(experiment_main_dir, experiment_id, model_summary):
    """
    Create a new model directory.
    :param experiment_main_dir: Where all experiments are stored.
    :param experiment_id: The ID of this experiment.
    :param model_summary: A summary string of the model.
    :return: A directory where we can store model logs. Raises an exception if the model directory already exists.
    """
    model_name = "{}-{}".format(experiment_id, model_summary)
    model_dir = os.path.join(experiment_main_dir, model_name)
    if os.path.exists(model_dir):
        raise ValueError("Model directory already exists {}".format(model_dir))
    os.makedirs(model_dir)
    return model_dir


def main(config):



    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])

    dataset = MyOwnDataset(root=config.train_data_dir, directory=config.train_data_dir,
                           transform=transform)
    loader = DataLoader(dataset, batch_size=config.bs_train)

    model = VGAE2(VariationalEncoder(config))
    model = model.to(C.DEVICE)
    experiment_id = int(time.time())
    experiment_name = model.model_name()
    model_dir = create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)

    #save config as json in model directory
    config.to_json(os.path.join(model_dir, 'config.json'))
    # Save the command line that was used.
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(cmd)

    writer = SummaryWriter(os.path.join(model_dir, 'logs'))
    optim_name = config.optimizer
    if optim_name == 'sgd':
        print('using sgd')
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    weight_decay=config.lr_decay,
                                    momentum=config.momentum,
                                    nesterov=config.nesterov)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.lr_decay)

    if config.scheduler == 'reduce_plateau':
        scheduler = optimizer.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=5, verbose=False)
    elif config.scheduler == 'step':
        scheduler = optimizer.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
    else:
        raise NotImplementedError

    #training loop
    global_step = 0
    for epoch in range(config.n_epochs):
        for i, abatch in tqdm(enumerate(loader)):
            print(f'Batch is in device {abatch.device}')
            start = time.time()
            optimizer.zero_grad()

            train_data, val_data, test_data = abatch
            train_losses = train(model)

            elapsed = time.time() - start
            #losses.append(loss)

            #writer.add_scalar('Loss/train', loss, global_step)
            for k in train_losses:
                prefix = '{}/{}'.format(k, 'train')
                writer.add_scalar(prefix, train_losses[k], global_step)

            writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], global_step)

            if global_step % (config.print_every - 1) == 0:
                loss_string = ' '.join(['{}: {:.6f}'.format(k, train_losses[k]) for k in train_losses])
                print('[TRAIN {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed))

            with open(os.path.join('/home/csolis/losses', f'losses_{model_name}.pkl'), 'ab') as f:
                pickle.dump(losses, f)
            # print(f'Loss: {loss:.4f}')
            if i % args.validation_steps == 0:
                # print(loss)
                auc, ap = test(test_data)
                aucs.append(auc)
                with open(os.path.join('/home/csolis/auc', f'auc_{model_name}.pkl'), 'ab') as f:
                    pickle.dump(aucs, f)
                writer.add_scalar('AUC/train/test', auc, it)
                # print(f'Iteration: {i:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        print(f'Loss of epochs last iteration: {loss}')
    print('Finished training.')
    print('Saving losses to dir')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--validation_steps', type=int, default=100)
    parser.add_argument('--default_cfg_path', default='/home/csolis/cfgs/default_config.yaml')
    args = parser.parse_args()
    print(f'Arguments:\n {args}')

    writer = SummaryWriter()

    config = load_config(args.default_cfg_path)
    print(f'config is {config}')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {C.DEVICE}')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = MyOwnDataset(root=config['data']['train_data_dir'], directory=config['data']['train_data_dir'], transform=transform)
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'])
    model_name = config.model_name

    in_channels, out_channels = dataset.num_features, config['model']['layers']['output_layer']
    kwargs = {'dropout': config['model']['dropout'], 'num_layers': args.num_layers}
    #model = VGAE(VariationalEncoder(config))


    #model = model.to(C.DEVICE)
    optim_name = config['optimizer']['name']
    if optim_name == 'sgd':
        print('using sgd')
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['optimizer']['lr'],
                                    weight_decay=config['optimizer']['weight_decay'],
                                    momentum=config['optimizer']['momentum'],
                                    nesterov=config['optimizer']['nesterov'])
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    losses = []
    aucs = []
    it = 0
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch: {epoch:03d}')
        for i, data in tqdm(enumerate(loader)):
            train_data, val_data, test_data = data
            it += 1
            loss = train()
            losses.append(loss)
            writer.add_scalar('Loss/train', loss, it)
            with open(os.path.join('/home/csolis/losses', f'losses_{model_name}.pkl'), 'ab') as f:
                pickle.dump(losses, f)
            #print(f'Loss: {loss:.4f}')
            if i % args.validation_steps == 0:
                #print(loss)
                auc, ap = test(test_data)
                aucs.append(auc)
                with open(os.path.join('/home/csolis/auc', f'auc_{model_name}.pkl'), 'ab') as f:
                    pickle.dump(aucs, f)
                writer.add_scalar('AUC/train/test', auc, it)
                #print(f'Iteration: {i:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        print(f'Loss of epochs last iteration: {loss}')
    print('Finished training.')
    print('Saving losses to dir')



    # get final embeddings
    transform_test = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])
    dataset_test = MyOwnDataset(root=args.dir_test_data, transform=transform_test)
    loader = DataLoader(dataset_test, batch_size=args.batch_size)
    model.eval()
    if os.path.exists(args.save_embeddings_dir) and os.path.isdir(args.save_embeddings_dir):
        shutil.rmtree(args.save_embeddings_dir)
        os.mkdir(args.save_embedding_dir)

    for i, data in enumerate(loader):
        z = model.encode(data.x, data.edge_index)
        torch.save(z, os.path.join(args.save_embeddings_dir, f'{args.encoder_type}', f'emb_{i:03d}.pt'))
    main(Configuration.parse_cmd())




