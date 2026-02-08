# sae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder adapted from:
    https://github.com/saprmarks/dictionary_learning

    This module learns a sparse dictionary over CLIP embeddings.
    """

    def __init__(
            self,
            input_dim: int,
            dict_size: int,
            l1_alpha: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.l1_alpha = l1_alpha

        self.encoder = nn.Linear(input_dim, dict_size, bias=True)

        self.decoder = nn.Linear(dict_size, input_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        nn.init.kaiming_uniform_(self.decoder.weight, a=0.0)
        with torch.no_grad():
            self.decoder.weight.data[:] = F.normalize(
                self.decoder.weight.data, dim=0
            )


        self.decoder.weight.requires_grad = False
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):

        return F.relu(self.encoder(x))  # [B, dict_size]

    def decode(self, z):

        return self.decoder(z)  # [B, input_dim]

    def forward(self, x):

        z = self.encode(x)
        recon = self.decode(z)

        l1_loss = self.l1_alpha * z.abs().sum(dim=1).mean()

        return recon, z, l1_loss
