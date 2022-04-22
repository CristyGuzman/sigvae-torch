import argparse
import os
import pickle
import shutil
from data import MyOwnDataset
import torch
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.autoencoder import VGAE
from encoders import VariationalEncoder
import yaml

#CONFIG_PATH = "./"
def load_config(config_path):
    #with open(os.path.join(CONFIG_PATH, config_name)) as file:
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config





def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--validation_steps', type=int, default=100)
    parser.add_argument('--default_cfg_path', default='/home/csolis/cfgs/default_config.yaml')
    args = parser.parse_args()
    print(f'Arguments:\n {args}')
    config = load_config(args.default_cfg_path)
    print(f'config is {config}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = MyOwnDataset(root=config['data']['train_data_dir'], directory=config['data']['train_data_dir'], transform=transform)
    dataset[0]
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'])
    model_name = config['model']['name']

    in_channels, out_channels = dataset.num_features, config['model']['layers']['output_layer']
    kwargs = {'dropout': config['model']['dropout'], 'num_layers': args.num_layers}
    model = VGAE(VariationalEncoder(in_channels, out_channels, encoder_type=model_name, **kwargs))


    model = model.to(device)
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
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch: {epoch:03d}')
        for i, data in tqdm(enumerate(loader)):
            train_data, val_data, test_data = data
            loss = train()
            losses.append(loss)
            #print(f'Loss: {loss:.4f}')
            if i % args.validation_steps == 0:
                #print(loss)
                auc, ap = test(test_data)
                aucs.append(auc)
                #print(f'Iteration: {i:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        print(f'Loss of epochs last iteration: {loss}')
    print('Finished training.')
    print('Saving losses to dir')
    with open(os.path.join('/home/csolis/losses', f'losses_{model_name}.pkl'), 'wb') as f:
        pickle.dump(losses)
    with open(os.path.join('/home/csolis/auc', f'auc_{model_name}.pkl'), 'wb') as f:
        pickle.dump(aucs)

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





