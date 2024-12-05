import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


def download_cifar10_data():
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    print("Datasets downloaded successfully.")
    return train_dataset, test_dataset


def prepare_cifar10_data(transform, batch_size=64, val_split=0.2):
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def show_reconstructed_images(model, test_loader, device, num_images=5):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)[:num_images]
    with torch.no_grad():
        outputs, _, _ = model(images)
    images = images.cpu()
    outputs = outputs.cpu()

    fig, axes = plt.subplots(nrows=2, ncols=num_images, sharex=True, sharey=True, figsize=(15, 4))
    for i in range(num_images):
        # Original images
        img = images[i]
        img = img.permute(1, 2, 0)
        axes[0, i].imshow(img.numpy())
        axes[0, i].axis('off')

        # Reconstructed images
        recon = outputs[i]
        recon = recon * 0.5 + 0.5  # Unnormalize
        recon = recon.permute(1, 2, 0)
        axes[1, i].imshow(recon.numpy())
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.show()


def train_model(writer, model, train_loader, val_loader, optimizer, compute_loss, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0

            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss = compute_loss(model, inputs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)

            if phase == 'val':
                model.eval()
                with torch.no_grad():
                    inputs, _ = next(iter(val_loader))
                    inputs = inputs.to(device)
                    outputs = model.get_reconstructed(inputs)

                    N = min(inputs.size(0), 4)
                    inputs = inputs[:N]
                    outputs = outputs[:N]

                    input_grid = torchvision.utils.make_grid(inputs.cpu())
                    output_grid = torchvision.utils.make_grid(outputs.cpu())

                    writer.add_image('Input Images', input_grid, epoch)
                    writer.add_image('Reconstructed Images', output_grid, epoch)

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_model.pth')

    return model


def evaluate_model(model, test_loader, compute_loss, device, writer):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            loss = compute_loss(model, inputs)

            running_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

    average_loss = running_loss / num_samples
    print(f'Test Loss: {average_loss:.4f}')

    writer.add_scalar('Loss/test', average_loss, global_step=0)

    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    outputs = model.get_reconstructed(inputs)

    N = min(inputs.size(0), 4)
    inputs = inputs[:N]
    outputs = outputs[:N]

    input_grid = torchvision.utils.make_grid(inputs.cpu())
    output_grid = torchvision.utils.make_grid(outputs.cpu())

    writer.add_image('Test/Input Images', input_grid, global_step=0)
    writer.add_image('Test/Reconstructed Images', output_grid, global_step=0)

    return average_loss
