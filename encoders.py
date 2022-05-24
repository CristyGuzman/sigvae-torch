import torch
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GINConv, global_add_pool, GATConv
import torch.nn.functional as F
import torch.distributions as tdist
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn.models.autoencoder import VGAE
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
from model import GraphDecoder

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GINEncoder(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hid_channels, out_channels):
        super(GINEncoder, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(GINConv(Sequential(
                Linear(self.in_channels, self.hid_channels), BatchNorm1d(self.hid_channels), ReLU(),
                           Linear(self.hid_channels, self.out_channels), ReLU())))
        else:
            self.layers.append(GINConv(Sequential(
                Linear(self.in_channels, self.hid_channels), BatchNorm1d(self.hid_channels), ReLU(),
                Linear(self.hid_channels, self.hid_channels), ReLU())))

        if num_layers > 2:
            for i in range(num_layers-2):
                self.layers.append(GINConv(
                Sequential(Linear(self.hid_channels, self.hid_channels), BatchNorm1d(self.hid_channels), ReLU(),
                           Linear(self.hid_channels, self.hid_channels), ReLU())))

        if self.num_layers > 1:
            self.layers.append(GINConv(
                Sequential(Linear(self.hid_channels, self.hid_channels), Linear(self.hid_channels, self.out_channels))))

    def forward(self, x, edge_index):
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)

        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, heads, out_channels, dropout):
        super(GATEncoder, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.heads = heads
        self.conv1 = GATConv(self.in_channels, self.hid_channels, heads=self.heads, dropout=self.dropout)
        self.conv2 = GATConv(self.hid_channels*self.heads, self.out_channels, concat=False, heads=1, dropout=self.dropout)


    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class VariationalEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_type = config.encoder_type
        self.in_channels = config.input_size
        self.out_channels = config.output_size
        if self.encoder_type == 'gcn':
            self.conv1 = GCNConv(self.in_channels, 2 * self.out_channels)
            self.conv2 = GCNConv(2 * self.out_channels, 2 * self.out_channels)
            self.conv_mu = GCNConv(2 * self.out_channels, self.out_channels)
            self.conv_logstd = GCNConv(2 * self.out_channels, self.out_channels)
        elif self.encoder_type == 'gin':
            if config.num_layers is None:
                raise ValueError('Number of layers were not provided')
            else:
                self.num_layers = config.num_layers
            self.conv1 = GINEncoder(self.num_layers, self.in_channels, 2 * self.out_channels, 2 * self.out_channels)
            self.conv_mu = GINEncoder(1, 2 * self.out_channels, 2 * self.out_channels, self.out_channels)
            self.conv_logstd = GINEncoder(1, 2 * self.out_channels, self.out_channels, self.out_channels)
        elif self.encoder_type == 'gat':
            if config.dropout is None:
                raise ValueError('Dropout not provided')
            else:
                self.dropout = config.dropout
            self.conv1 = GATEncoder(in_channels=self.in_channels, hid_channels=2*self.out_channels, heads=5, out_channels=2*self.out_channels, dropout=self.dropout)
            self.conv_mu = GATEncoder(in_channels=2*self.out_channels, hid_channels=2*self.out_channels, heads=5,
                                   out_channels=self.out_channels, dropout=self.dropout)
            self.conv_logstd = GATEncoder(in_channels=2 * self.out_channels, hid_channels=2 * self.out_channels, heads=5,
                                      out_channels=self.out_channels, dropout=self.dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VGAE2(VGAE):
    def __init__(self, config, encoder, decoder=None):
        super(VGAE2, self).__init__(encoder, decoder)
        self.config = config

    def model_name(self):
        """A summary string of this model. Override this if desired."""
        return '{}-{}'.format(self.__class__.__name__, self.config.tag)
    def encode(self, *args, **kwargs):
        mu, sigma = self.encoder(*args, **kwargs)
        emb_mu = mu[self.K:, :]
        emb_logvar = sigma[self.K:, :]

        # check tensor size compatibility
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'

        z, eps = self.reparametrize(emb_mu, emb_logvar)

    def reconstruction_loss(self, z, pos_edge_index, neg_edge_index, all_edge_index):
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        if neg_edge_index is None:
            # Do not include self-loops in negative samples
            all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
            all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)
            neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return y, pred

class SIGVAE(torch.nn.Module):
    def __init__(self, config):
        super(SIGVAE, self).__init__()
        ## Follow layer definition as in original SIGVAE implementation
        self.config = config
        self.in_channels = config.input_size
        self.out_channels = config.output_size
        self.e_channels = config.noise_size
        self.hidden_dim = config.hidden_size
        self.gcn_e = GCNConv(self.e_channels, self.hidden_dim)
        self.gcn_1 = GCNConv(self.in_channels, self.hidden_dim)
        self.gcn_mu = GCNConv(self.hidden_dim, self.out_channels)
        self.gcn_sigma = GCNConv(self.hidden_dim, self.out_channels)
        self.dc = GraphDecoder(self.out_channels, config.dropout, gdc='ip')
        if config.ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5]))
        elif config.ndist == 'Normal':
            self.ndist = tdist.Normal(torch.tensor([0.]), torch.tensor([1.]))
        elif config.ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.]))
        self.K = config.K
        self.J = config.copyJ

        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).
        self.reweight = ((self.e_channels + self.hidden_dim) / (self.in_channels + self.hidden_dim)) ** (.5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps

    def forward(self, x, edge_index):
        hidden_x = self.gcn_1(x, edge_index).relu()
        e = self.ndist.sample(torch.Size([self.K + self.J, x.shape[1], self.e_channels]))
        e = torch.squeeze(e, -1)
        e = e.mul(self.reweight)
        hidden_e = self.gcn_e(e, edge_index)

        hidden1 = hidden_x + hidden_e
        p_signal = hidden_x.pow(2.).mean()
        p_noise = hidden_e.pow(2.).mean([-2, -1])
        snr = (p_signal / p_noise)
        mu = self.gcn_mu(hidden1, edge_index)
        sigma = self.gcn_sigma(hidden1, edge_index)
        emb_mu = mu[self.K:, :]
        emb_logvar = sigma[self.K:, :]

        # check tensor size compatibility
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'

        z, eps = self.reparameterize(emb_mu, emb_logvar)

        adj_, z_scaled, rk = self.dc(z)

        return adj_, mu, sigma, z, z_scaled, eps, rk, snr

        #return mu, sigma, z, eps



        #return self.conv2(x, edge_index)