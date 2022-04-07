from torch_geometric.nn import VGAE
import torch

class SIGVAE(VGAE):
    def __init__(self, encoder, decoder):
        super(SIGVAE, self).__init__(decoder, encoder)

    def encode(self, *args, **kwargs):
        mu, logvar, snr = self.encoder(*args, **kwargs)

        emb_mu = mu[self.K:, :]
        emb_logvar = logvar[self.K:, :]

        # check tensor size compatibility
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'

        std = torch.exp(emb_logvar / 2.)
        eps = torch.randn_like(std)
        z, eps = eps.mul(std).add(emb_mu), eps
        return z, eps, mu, logvar, snr

    def forward(self, *args, **kwargs):
        z, eps, mu, logvar, snr = self.encode(*args, **kwargs)
        adj_, z_scaled, rk = self.dc(z)

        return adj_, mu, logvar, z, z_scaled, eps, rk, snr

    def decode(self, z, eps):
        pass


