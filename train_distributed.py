"""Serial simulation of synchronous distributed SGD with mean aggregation."""

import argparse
import time
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
from experiment_logger import ExperimentRecorder, get_git_metadata
from model import SimpleCNN
from train_baseline import evaluate


def train_one_distributed_epoch(
    model: nn.Module,
    worker_loaders: list[torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_rounds: int | None = None,
    recorder: ExperimentRecorder | None = None,
    epoch: int = 1,
    global_round_start: int = 0,
    test_loader: torch.utils.data.DataLoader | None = None,
    eval_interval_rounds: int = 0,
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
    for round_in_epoch, worker_batches in enumerate(progress, start=1):
        round_started = time.perf_counter()
        worker_gradients = []
        round_loss = 0.0
        round_correct = 0
        round_examples = 0
        for batch in worker_batches:
            gradient, loss, correct, batch_size = compute_worker_gradient(
                model, batch, criterion, device
            )
            worker_gradients.append(gradient)
            round_loss += loss * batch_size
            round_correct += correct
            round_examples += batch_size
            total_loss += loss * batch_size
            total_correct += correct
            total_examples += batch_size

        stacked_gradients = torch.stack(worker_gradients)
        if stacked_gradients.is_cuda:
            aggregation_start = torch.cuda.Event(enable_timing=True)
            aggregation_end = torch.cuda.Event(enable_timing=True)
            aggregation_start.record()
            global_gradient = mean_aggregate(stacked_gradients)
            aggregation_end.record()
            aggregation_end.synchronize()
            aggregation_time = aggregation_start.elapsed_time(aggregation_end) / 1000.0
        else:
            aggregation_started = time.perf_counter()
            global_gradient = mean_aggregate(stacked_gradients)
            aggregation_time = time.perf_counter() - aggregation_started

        # The server applies exactly one update after collecting all workers.
        optimizer.zero_grad(set_to_none=True)
        assign_flat_gradient(model, global_gradient)
        optimizer.step()

        if recorder is not None:
            global_round = global_round_start + round_in_epoch
            recorder.record_round(
                global_round=global_round,
                epoch=epoch,
                round_in_epoch=round_in_epoch,
                train_loss=round_loss / round_examples,
                train_accuracy=round_correct / round_examples,
                gradient_norm=global_gradient.norm().item(),
                round_time_seconds=time.perf_counter() - round_started,
                aggregation_time_seconds=aggregation_time,
            )
            recorder.record_aggregation(
                global_round=global_round,
                aggregator="mean",
                selected_worker_ids=range(len(worker_gradients)),
            )

            should_evaluate = (
                test_loader is not None
                and eval_interval_rounds > 0
                and global_round % eval_interval_rounds == 0
                and round_in_epoch < round_limit
            )
            if should_evaluate:
                test_loss, test_accuracy = evaluate(
                    model, test_loader, criterion, device
                )
                recorder.record_metric(
                    global_round=global_round,
                    epoch=epoch,
                    split="test",
                    loss=test_loss,
                    accuracy=test_accuracy,
                )
                model.train()

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
    parser.add_argument("--seed", type=int, default=0, help="model and training seed")
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="optional rounds per epoch for quick verification",
    )
    parser.add_argument(
        "--eval-interval-rounds",
        type=int,
        default=0,
        help="evaluate every N global rounds; 0 means epoch-end only",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="disable writing experiment files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.max_rounds is not None and args.max_rounds <= 0:
        raise ValueError("--max-rounds must be positive")
    if args.eval_interval_rounds < 0:
        raise ValueError("--eval-interval-rounds cannot be negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    worker_loaders, test_loader = get_worker_loaders(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
        partition_seed=args.partition_seed,
    )
    print(f"Workers: {len(worker_loaders)}")
    print(f"Samples per worker: {[len(loader.dataset) for loader in worker_loaders]}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.learning_rate)

    recorder = None
    if not args.no_record:
        config = {
            "dataset": "CIFAR-10",
            "model": "SimpleCNN",
            "aggregator": "mean",
            "num_workers": args.num_workers,
            "num_byzantine": 0,
            "byzantine_worker_ids": [],
            "attack": "none",
            "attack_parameters": {},
            "multi_krum_m": None,
            "epochs": args.epochs,
            "max_rounds_per_epoch": args.max_rounds,
            "batch_size_per_worker": args.batch_size,
            "nominal_effective_batch_size": args.batch_size * args.num_workers,
            "learning_rate": args.learning_rate,
            "optimizer": "SGD",
            "partition_seed": args.partition_seed,
            "training_seed": args.seed,
            "evaluation_interval_rounds": args.eval_interval_rounds,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **get_git_metadata(),
        }
        recorder = ExperimentRecorder(
            config=config,
            results_dir=args.results_dir,
            run_name=args.run_name,
        )
        print(f"Results: {recorder.run_dir}")

    global_round = 0
    final_test_loss = None
    final_test_accuracy = None
    try:
        if recorder is not None:
            initial_test_loss, initial_test_accuracy = evaluate(
                model, test_loader, criterion, device
            )
            recorder.record_metric(
                global_round=0,
                epoch=0,
                split="test",
                loss=initial_test_loss,
                accuracy=initial_test_accuracy,
            )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy, rounds = train_one_distributed_epoch(
                model,
                worker_loaders,
                criterion,
                optimizer,
                device,
                max_rounds=args.max_rounds,
                recorder=recorder,
                epoch=epoch,
                global_round_start=global_round,
                test_loader=test_loader,
                eval_interval_rounds=args.eval_interval_rounds,
            )
            global_round += rounds
            final_test_loss, final_test_accuracy = evaluate(
                model, test_loader, criterion, device
            )

            if recorder is not None:
                recorder.record_metric(
                    global_round=global_round,
                    epoch=epoch,
                    split="train_epoch",
                    loss=train_loss,
                    accuracy=train_accuracy,
                )
                recorder.record_metric(
                    global_round=global_round,
                    epoch=epoch,
                    split="test",
                    loss=final_test_loss,
                    accuracy=final_test_accuracy,
                )

            print(f"Epoch {epoch}/{args.epochs} ({rounds} rounds)")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
            print(f"Test Loss: {final_test_loss:.4f}")
            print(f"Test Accuracy: {final_test_accuracy * 100:.2f}%")

        if recorder is not None:
            recorder.finalize(
                completed=True,
                total_rounds=global_round,
                final_test_loss=final_test_loss,
                final_test_accuracy=final_test_accuracy,
                final_test_error=(
                    None
                    if final_test_accuracy is None
                    else 1.0 - final_test_accuracy
                ),
            )
    except Exception as error:
        if recorder is not None:
            recorder.finalize(
                completed=False,
                last_completed_round=recorder.last_global_round,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        raise
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
