"""Serial simulation of synchronous distributed SGD with mean aggregation."""

import argparse
from itertools import islice

import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from data import get_worker_loaders
from distributed import (
    assign_flat_gradient,
    compute_worker_gradient,
    mean_aggregate,
)
from model import SimpleCNN
from train_baseline import evaluate


def train_one_distributed_epoch(
    model: nn.Module,
    worker_loaders: list[torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_rounds: int | None = None,
) -> tuple[float, float, int]:
    """Run synchronous rounds, computing every worker serially per round."""
    model.train()
    available_rounds = min(len(loader) for loader in worker_loaders)
    round_limit = available_rounds if max_rounds is None else min(max_rounds, available_rounds)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    batches_by_round = islice(
        zip(*(iter(loader) for loader in worker_loaders)), round_limit
    )
    progress = tqdm(batches_by_round, total=round_limit, desc="Rounds", leave=False)
    for worker_batches in progress:
        worker_gradients = []
        for batch in worker_batches:
            gradient, loss, correct, batch_size = compute_worker_gradient(
                model, batch, criterion, device
            )
            worker_gradients.append(gradient)
            total_loss += loss * batch_size
            total_correct += correct
            total_examples += batch_size

        stacked_gradients = torch.stack(worker_gradients)
        global_gradient = mean_aggregate(stacked_gradients)

        # The server applies exactly one update after collecting all workers.
        optimizer.zero_grad(set_to_none=True)
        assign_flat_gradient(model, global_gradient)
        optimizer.step()

    return total_loss / total_examples, total_correct / total_examples, round_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 with serial distributed SGD and Mean aggregation"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128, help="batch size per worker")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="optional rounds per epoch for quick verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.max_rounds is not None and args.max_rounds <= 0:
        raise ValueError("--max-rounds must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    worker_loaders, test_loader = get_worker_loaders(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
    )
    print(f"Workers: {len(worker_loaders)}")
    print(f"Samples per worker: {[len(loader.dataset) for loader in worker_loaders]}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, rounds = train_one_distributed_epoch(
            model,
            worker_loaders,
            criterion,
            optimizer,
            device,
            max_rounds=args.max_rounds,
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} ({rounds} rounds)")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
