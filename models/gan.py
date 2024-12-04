import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
import torchvision.utils as vutils


# Generator and Discriminator classes remain the same
class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size. 1
        )

    def forward(self, input):
        return self.main(input)


# Corrected FID calculation functions
def calculate_activation_statistics(images, model, cuda=False):
    model.eval()
    with torch.no_grad():
        if cuda:
            images = images.cuda()
        else:
            images = images.cpu()
        # Resize images to (299, 299)
        images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize images to [-1, 1]
        images_normalized = 2 * images_resized - 1
        pred = model(images_normalized)['pool']  # 'pool' is the layer name for the final average pooling
        act = pred.cpu().numpy().reshape(pred.size(0), -1)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Frechet Distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Compute square root of product of covariance matrices
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # Numerical stability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Avoid imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid


def calculate_fretchet(real_images, fake_images, model, cuda=False):
    mu1, sigma1 = calculate_activation_statistics(real_images, model, cuda=cuda)
    mu2, sigma2 = calculate_activation_statistics(fake_images, model, cuda=cuda)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def bce_loss(predictions, targets):
    eps = 1e-7  # Small value to avoid log(0)
    predictions = torch.clamp(predictions, eps, 1 - eps)  # Clamp predictions to avoid instability
    return -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)).mean()


def train_gan(netG, netD, dataloader, criterion, optimizerG, optimizerD, feature_extractor, device,
              num_epochs, nz, fixed_noise, writer, real_label=1.0, fake_label=0.0):
    """Train the GAN model."""
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            # Add noise to real images
            real_cpu = 0.9 * real_cpu + 0.1 * torch.randn_like(real_cpu, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Generate fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            # Add noise to fake images
            fake = 0.9 * fake + 0.1 * torch.randn_like(fake, device=device)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            # Update D
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # Reverse labels for generator training
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Log losses to TensorBoard
            writer.add_scalar('Generator Loss', errG.item(), iters)
            writer.add_scalar('Discriminator Loss', errD.item(), iters)

            # Save generated images for visualization and log to TensorBoard
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_display = netG(fixed_noise).detach().cpu()
                grid = vutils.make_grid(fake_display, padding=2, normalize=True)
                img_list.append(grid)
                writer.add_image('Generated Images', grid, iters)

            iters += 1

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Calculate FID
        fretchet_dist = calculate_fretchet(real_cpu, fake, feature_extractor, cuda=device.type == 'cuda')

        # Log FID to TensorBoard
        writer.add_scalar('FID', fretchet_dist, epoch)

        if (epoch + 1) % 5 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f' %
                  (epoch + 1, num_epochs, errD.item(), errG.item(), fretchet_dist))
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            idx = torch.randint(len(fake_display), (10,))
            pictures = vutils.make_grid(fake_display[idx], nrow=5, padding=2, normalize=True)
            plt.imshow(np.transpose(pictures, (1, 2, 0)))
            plt.show()

    return img_list, G_losses, D_losses
