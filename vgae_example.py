import argparse
import collections
import os
import pickle
import shutil
from data import MyOwnDataset
import torch
import torch.optim as optim
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
#from torch_geometric.nn.models.autoencoder import VGAE
from encoders import VariationalEncoder, VGAE2
import yaml
from torch.utils.tensorboard import SummaryWriter
from configuration import CONSTANTS as C
from configuration import Configuration
import time
from torch import nn
import sys
import numpy as np
from metrics import Metrics

#CONFIG_PATH = "./"
def load_config(config_path):
    #with open(os.path.join(CONFIG_PATH, config_name)) as file:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config



def get_losses(model, z, data):
    recon_loss = model.recon_loss(z, data.pos_edge_label_index)
    kl_loss = model.kl_loss(model.__mu__, model.__logstd__)
    print(kl_loss)
    loss = recon_loss + (1 / data.num_nodes) * kl_loss
    return loss, {'total_loss': float(loss), 'recon_loss': float(recon_loss), 'kl_loss': float(kl_loss)}

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    #recon_loss = model.recon_loss(z, data.pos_edge_label_index)
    loss, losses = get_losses(model, z, data)
    #loss = losses['total_loss']
    #kl_loss = model.kl_loss
    #loss = recon_loss + (1 / train_data.num_nodes) * kl_loss
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
    optimizer.step()
    return losses

@torch.no_grad()
def test(model, data, metrics_engine):
    loss_vals_agg = collections.defaultdict(float)
    n_samples = 0
    #model.eval()
    for i, abatch in tqdm(enumerate(data)):
        _, valid_data, _ = abatch
        print(f"shape is {valid_data.shape}")
        z = model.encode(valid_data.x, valid_data.edge_index)
        _, losses = get_losses(model, z, valid_data)
        for k in losses:
            loss_vals_agg += losses[k]*valid_data.batch_size
        targets, preds = model.test(z, valid_data.pos_edge_label_index, valid_data.neg_edge_label_index)
        metrics_engine.compute_and_aggregate(preds, targets)
        n_samples += valid_data.batch_size

    for k in loss_vals_agg:
        loss_vals_agg[k] /= n_samples

    return loss_vals_agg

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

def to_tensorboard_log(metrics, writer, global_step, prefix=''):
    """Write metrics to tensorboard."""
    for m in metrics:
        writer.add_scalar('{}/{}'.format(m, prefix), metrics[m], global_step)



def main(config):

    print(C)
    device = C.DEVICE

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(C.DEVICE),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    print('Creating training dataset')
    dataset = MyOwnDataset(root=config.train_data_dir, directory=config.train_data_dir,
                           transform=transform)
    print('Creating validation dataset')
    valid_dataset = MyOwnDataset(root=config.valid_data_dir, directory=config.valid_data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=config.bs_train)
    valid_loader = DataLoader(valid_dataset, batch_size=config.bs_eval)
    print(f"next iter is {next(iter(valid_loader))}")
    me = Metrics()
    if config.input_size is None:
        config.input_size = dataset.num_features
    model = VGAE2(config=config, encoder=VariationalEncoder(config=config))
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=5, verbose=False)
    elif config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
    else:
        raise NotImplementedError

    #training loop
    global_step = 0
    for epoch in range(config.n_epochs):
        for i, abatch in tqdm(enumerate(loader)):

            start = time.time()
            #optimizer.zero_grad()

            train_data, val_data, test_data = abatch
            #print(f'Batch is in device {train_data.device}')
            train_losses = train(model, optimizer, train_data)

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

            if global_step % (config.eval_every - 1) == 0:
                # Evaluate on validation.
                start = time.time()
                model.eval()
                valid_losses = test(model, valid_loader, me)
                valid_metrics = me.get_final_metrics()
                elapsed = time.time() - start

                # Log to console.
                loss_string = ' '.join(['{}: {:.6f}'.format(k, valid_losses[k]) for k in valid_losses])
                print('[VALID {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed))
                s = "Eval metrics: \n"
                for m in sorted(valid_metrics):
                    val = np.sum(valid_metrics[m])
                    s += "   {}: {:.3f}".format(m, val)

                print('[VALID {:0>5d} | {:0>3d}] {}'.format(i + 1, epoch + 1, s))

                # Log to tensorboard.
                to_tensorboard_log(valid_losses, writer, global_step, 'valid')
                to_tensorboard_log(valid_metrics, writer, global_step, 'valid')
            global_step += 1
        scheduler.step(epoch)

            #with open(os.path.join('/home/csolis/losses', f'losses_{model_name}.pkl'), 'ab') as f:
             #   pickle.dump(losses, f)
            # print(f'Loss: {loss:.4f}')
            #if i % args.validation_steps == 0:
                # print(loss)
             #   auc, ap = test(test_data)
              #  aucs.append(auc)
               # with open(os.path.join('/home/csolis/auc', f'auc_{model_name}.pkl'), 'ab') as f:
                 #   pickle.dump(aucs, f)
                #writer.add_scalar('AUC/train/test', auc, it)
                # print(f'Iteration: {i:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        #print(f'Loss of epochs last iteration: {loss}')
    print('Finished training.')
    #print('Saving losses to dir')




if __name__ == '__main__':
    main(Configuration.parse_cmd())




