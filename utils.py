import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split


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


def train_model(model, train_loader, val_loader, optimizer, compute_loss, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Each epoch has a training and validation phase
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
                    # Compute loss using the provided compute_loss function
                    loss = compute_loss(model, inputs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')

    return model


def evaluate_model(model, test_loader, compute_loss, device):
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            loss = compute_loss(model, inputs)

            running_loss += loss.item()
            num_samples += inputs.size(0)

    average_loss = running_loss / num_samples
    print(f'Test Loss: {average_loss:.4f}')
    return average_loss
