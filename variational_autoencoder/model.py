import torch
import torch.nn as nn
import numpy as np

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Sequence


class VAE(nn.Module):
    def __init__(self, shape_input: Union[Sequence, int], encoder_units: Optional[list] = None,
                 decoder_units: Optional[list] = None, latent_dimension: int = 5):
        super().__init__()

        self.shape = shape_input
        self.shape_flattened: int = np.prod(shape_input, axis=None)  # noqa

        self.encoder_units = encoder_units or [400]
        self.decoder_units = decoder_units or [400]

        def get_sequential(unit_list: List[int]) -> nn.Sequential:
            module_list = []
            for in_features, out_features in zip(unit_list, unit_list[1:]):
                module_list.append(nn.Linear(in_features, out_features))
                module_list.append(nn.ReLU())
                module_list.append(nn.BatchNorm1d(out_features))
            return nn.Sequential(*module_list)

        self.encoder = get_sequential([self.shape_flattened] + self.encoder_units)
        self.encoder_mu = nn.Linear(self.encoder_units[-1], latent_dimension)
        self.encoder_log_sigma = nn.Linear(self.encoder_units[-1], latent_dimension)

        self.decoder = get_sequential([latent_dimension] + self.decoder_units)
        self.decoder_out = nn.Linear(self.decoder_units[-1], self.shape_flattened)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flattened = x.view(-1, self.shape_flattened)
        x_encoded = self.encoder(x_flattened)
        return self.encoder_mu(x_encoded), self.encoder_log_sigma(x_encoded)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return torch.sigmoid(self.decoder_out(x)).view(-1, *self.shape)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        return self.decode(z), mu, log_sigma

    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.shape_flattened)  # maybe not keep
        x_reconstructed, mu, log_sigma = self(x)
        reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')  # noqa
        # reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')  # noqa
        kl_divergence = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return reconstruction_loss + kl_divergence

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.shape_flattened)  # maybe not keep
        mu, log_sigma = self.encode(x)
        return self.reparameterize(mu, log_sigma)
