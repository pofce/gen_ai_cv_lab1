import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.matmul(z, self.w.t()) + self.b  # [batch_size, 1]
        activation = torch.tanh(linear)
        z_new = z + self.u * activation  # [batch_size, dim]
        return z_new, self._log_det_jacobian(z, activation)

    def _log_det_jacobian(self, z, activation):
        psi = (1 - activation ** 2) * self.w  # Derivative of tanh(linear)
        det_jacobian = 1 + torch.matmul(psi, self.u.t())  # [batch_size, 1]
        return torch.log(torch.abs(det_jacobian) + 1e-6).squeeze()


class NFVAE(nn.Module):
    def __init__(self, latent_dim=128, flow_length=2):
        super(NFVAE, self).__init__()
        self.latent_dim = latent_dim
        self.flow_length = flow_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Normalizing Flows
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(flow_length)])

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z_0 = self.reparameterize(mu, logvar)

        log_det_jacobian = 0
        z = z_0
        for flow in self.flows:
            z, ldj = flow(z)
            log_det_jacobian += ldj

        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 128, 4, 4)
        decoded = self.decoder(decoder_input)
        return decoded, mu, logvar, z_0, z, log_det_jacobian

    def generate(self, num_images=5, device='cuda'):
        z_0 = torch.randn(num_images, self.latent_dim).to(device)

        # Apply flows to z_0 to get z_K
        z = z_0
        for flow in self.flows:
            z, _ = flow(z)

        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 128, 4, 4)
        generated_imgs = self.decoder(decoder_input)

        imgs = (generated_imgs.detach().cpu() + 1) / 2  # from [-1, 1] to [0, 1]

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[i].imshow(imgs[i].permute(1, 2, 0).numpy())
            axes[i].axis('off')
        plt.show()

        return generated_imgs

    def get_reconstructed(self, inputs):
        recon_x, _, _, _, _, _ = self(inputs)
        return recon_x


def custom_mse_loss(recon_x, x):
    return torch.sum((recon_x - x) ** 2)


def vae_flow_loss_function(recon_x, x, mu, logvar, z0, zk, log_det_jacobian, beta=1.0):
    reconstruction_loss = custom_mse_loss(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence -= torch.sum(log_det_jacobian)
    return (reconstruction_loss + beta * kl_divergence) / x.size(0)


def compute_loss_vae_flow(model, inputs):
    recon_x, mu, logvar, z0, zk, log_det_jacobian = model(inputs)
    loss = vae_flow_loss_function(recon_x, inputs, mu, logvar, z0, zk, log_det_jacobian)
    return loss
