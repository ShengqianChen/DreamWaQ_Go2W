import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self,
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 decoder_hidden_dims = [64, 128],
                 sigma_min = 0.0,
                 sigma_max = 5.0):
        super(VAE, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Build Encoder
        self.encoder = MLPHistoryEncoder(
            num_obs = num_obs,
            num_history = num_history,
            num_latent = num_latent * 4,
            activation = activation,
            adaptation_module_branch_hidden_dims=[128],
        )

        self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        self.latent_var = nn.Linear(num_latent * 4, num_latent)

        self.vel_mu = nn.Linear(num_latent * 4, 3)
        self.vel_var = nn.Linear(num_latent * 4, 3)

        # Build Decoder
        modules = []
        activation_fn = get_activation(activation)
        decoder_input_dim = num_latent + 3
        modules.extend(
            [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
            activation_fn]
            )
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l],num_obs))
            else:
                modules.append(nn.Linear(decoder_hidden_dims[l],decoder_hidden_dims[l + 1]))
                modules.append(activation_fn)
        self.decoder = nn.Sequential(*modules)

    def encode(self,obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        vel_mu = self.vel_mu(encoded)
        vel_var = self.vel_var(encoded)

        # Apply constraints directly to logvar to ensure sigma stays within bounds
        latent_var = self._constrain_logvar(latent_var)
        vel_var = self._constrain_logvar(vel_var)

        return [latent_mu, latent_var, vel_mu, vel_var]

    def decode(self,z,v):
        input = torch.cat([z,v], dim = 1)
        output = self.decoder(input)
        return output

    def forward(self,obs_history):
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
        z = self.reparameterize(latent_mu, latent_var, self.sigma_min, self.sigma_max)
        vel = self.reparameterize(vel_mu, vel_var, self.sigma_min, self.sigma_max)
        return [z,vel],[latent_mu, latent_var, vel_mu, vel_var]

    def _constrain_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        """
        Constrain logvar to ensure sigma stays within [sigma_min, sigma_max] bounds.
        This is applied directly during encoding to ensure sigma is always bounded.

        Args:
            logvar: Log variance tensor

        Returns:
            Constrained logvar tensor
        """
        # Convert to sigma, apply constraints, then convert back to logvar
        sigma = torch.exp(0.5 * logvar)
        sigma_constrained = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        logvar_constrained = 2 * torch.log(sigma_constrained + 1e-8)  # Add small epsilon for numerical stability

        return logvar_constrained

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                      sigma_min: float = 0.0, sigma_max: float = 5.0) -> torch.Tensor:
        """
        Reparameterization trick. Note: logvar is already constrained in encode().

        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Log variance of the latent Gaussian (already constrained)
        :param sigma_min: (float) Minimum allowed standard deviation (for compatibility)
        :param sigma_max: (float) Maximum allowed standard deviation (for compatibility)
        :return: (Tensor) Sampled latent vector
        """
        sigma = torch.exp(0.5 * logvar)
        # Sample from standard normal distribution
        eps = torch.randn_like(sigma)

        # Reparameterization: z = μ + σ * ε
        return eps * sigma + mu

    def loss_fn(self, obs_history, next_obs, vel_target, kld_weight = 1.0):
        estimation, latent_params = self.forward(obs_history)
        z, v = estimation
        latent_mu, latent_var, vel_mu, vel_var = latent_params

        assert not torch.isnan(vel_target).any(), "vel_target contains NaN values"
        assert not torch.isinf(vel_target).any(), "vel_target contains Inf values"

        # Reconstruction next_obs loss
        recons = self.decode(z,vel_target)
        next_obs_without_command = next_obs.clone()
        next_obs_without_command[:, 6:9] = 0
        recons_loss = F.mse_loss(recons, next_obs_without_command, reduction='none').mean(-1)

        # Supervised loss
        vel_loss = F.mse_loss(v, vel_target, reduction='none').mean(-1)

        # KL divergence loss (logvar is already constrained in encode())
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=1)

        loss = recons_loss + vel_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'vel_loss': vel_loss,
            'kld_loss': kld_loss,
        }


    def sample(self,obs_history):
        estimation, _ = self.forward(obs_history)
        return estimation

    def inference(self,obs_history):
        _, latent_params = self.forward(obs_history)
        latent_mu, latent_var, vel_mu, vel_var = latent_params
        return [latent_mu, vel_mu]

    def get_constrained_latent_params(self, obs_history):
        """
        Get the constrained latent parameters for monitoring and debugging.
        Note: logvar is already constrained in encode(), so sigma is guaranteed to be in bounds.

        Returns:
            dict: Dictionary containing constrained mean and std for both latent and velocity
        """
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)

        # logvar is already constrained, so sigma is guaranteed to be in bounds
        latent_sigma = torch.exp(0.5 * latent_var)
        vel_sigma = torch.exp(0.5 * vel_var)

        return {
            'latent_mu': latent_mu,
            'latent_sigma': latent_sigma,
            'vel_mu': vel_mu,
            'vel_sigma': vel_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max
        }

class MLPHistoryEncoder(nn.Module):

    def __init__(self,
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 adaptation_module_branch_hidden_dims = [128],):
        super(MLPHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        input_size = num_obs * num_history
        output_size = num_latent

        activation = get_activation(activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], output_size))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.encoder = nn.Sequential(*adaptation_module_layers)

    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T * obs_dim)
        """
        output = self.encoder(obs_history)
        return output


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
