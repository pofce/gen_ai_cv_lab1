import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 128, 4, 4)
        decoded = self.decoder(decoder_input)
        return decoded, mu, logvar

    def generate(self, num_images=5, device='cuda'):
        """
        Generate multiple images and display them.
        """
        z = torch.randn(num_images, self.latent_dim).to(device)
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 128, 4, 4)
        generated_imgs = self.decoder(decoder_input)

        imgs = (generated_imgs.detach().cpu() + 1) / 2  # Rescale from [-1, 1] to [0, 1]

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[i].imshow(imgs[i].permute(1, 2, 0).numpy())
            axes[i].axis('off')
        plt.show()

        return generated_imgs

    def get_reconstructed(self, inputs):
        recon_x, _, _ = self(inputs)
        return recon_x


def custom_mse_loss(recon_x, x):
    return torch.sum((recon_x - x) ** 2)


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    reconstruction_loss = custom_mse_loss(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (reconstruction_loss + beta * kl_divergence) / x.size(0)


def compute_loss_vae(model, inputs):
    recon_x, mu, logvar = model(inputs)
    loss = vae_loss_function(recon_x, inputs, mu, logvar)
    return loss
