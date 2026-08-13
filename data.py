"""CIFAR-10 data loading for the single-machine baseline."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Per-channel statistics commonly used to normalize CIFAR-10 images.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    data_dir: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Return training and test loaders, downloading CIFAR-10 if necessary."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
