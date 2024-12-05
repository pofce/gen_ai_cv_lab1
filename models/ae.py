import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstructed(self, inputs):
        return self(inputs)


def compute_loss_autoencoder(model, inputs):
    outputs = model(inputs)
    loss = ((outputs - inputs) ** 2).mean()
    return loss
