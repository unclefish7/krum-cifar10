"""CIFAR-10 data loading for the single-machine baseline."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
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


def get_worker_loaders(
    data_dir: str | Path = "./data",
    num_workers: int = 10,
    batch_size: int = 128,
    loader_workers: int = 0,
    partition_seed: int = 0,
) -> tuple[list[DataLoader], DataLoader]:
    """Return IID worker loaders and the shared CIFAR-10 test loader.

    A fixed random permutation partitions the training set into equally sized,
    disjoint shards. With CIFAR-10's 50,000 samples and 10 workers, each worker
    receives exactly 5,000 samples.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

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

    base_size, remainder = divmod(len(train_dataset), num_workers)
    shard_sizes = [base_size + (worker_id < remainder) for worker_id in range(num_workers)]
    generator = torch.Generator().manual_seed(partition_seed)
    worker_datasets = random_split(train_dataset, shard_sizes, generator=generator)

    worker_loaders = [
        DataLoader(
            worker_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=loader_workers,
            pin_memory=True,
        )
        for worker_dataset in worker_datasets
    ]
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
    )
    return worker_loaders, test_loader
