"""Gradient operations for serial distributed SGD simulation."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AggregationResult:
    """Result shared by Mean, Krum, Multi-Krum and future rules."""

    gradient: Tensor
    selected_worker_ids: Tensor
    scores: Tensor | None = None


def flatten_gradient(model: nn.Module) -> Tensor:
    """Flatten the gradients of all trainable model parameters."""
    gradients = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError("A trainable parameter has no gradient")
        gradients.append(parameter.grad.detach().reshape(-1))
    return torch.cat(gradients).clone()


def compute_worker_gradient(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[Tensor, float, int, int]:
    """Compute one worker's gradient without updating the global model."""
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    gradient = flatten_gradient(model)
    correct = (logits.argmax(dim=1) == labels).sum().item()
    return gradient, loss.item(), correct, labels.size(0)


def mean_aggregate(worker_gradients: Tensor) -> Tensor:
    """Average a [num_workers, num_parameters] gradient tensor."""
    if worker_gradients.ndim != 2 or worker_gradients.size(0) == 0:
        raise ValueError("worker_gradients must have shape [num_workers, num_parameters]")
    return worker_gradients.mean(dim=0)


def _krum_scores(
    worker_gradients: Tensor,
    num_byzantine: int,
) -> Tensor:
    """Compute the Krum score of every submitted worker gradient.

    For each worker, the Krum score is the sum of squared distances to its
    ``num_workers - num_byzantine - 2`` closest other worker gradients.
    """
    if worker_gradients.ndim != 2 or worker_gradients.size(0) == 0:
        raise ValueError("worker_gradients must have shape [num_workers, num_parameters]")
    if num_byzantine < 0:
        raise ValueError("num_byzantine cannot be negative")

    num_workers = worker_gradients.size(0)
    if 2 * num_byzantine + 2 >= num_workers:
        raise ValueError(
            "Krum requires 2 * num_byzantine + 2 < num_workers, "
            f"got num_byzantine={num_byzantine}, num_workers={num_workers}"
        )

    # Compute the squared Euclidean distance matrix without materializing a
    # [num_workers, num_workers, num_parameters] intermediate tensor.
    squared_norms = worker_gradients.square().sum(dim=1, keepdim=True)
    squared_distances = (
        squared_norms
        + squared_norms.transpose(0, 1)
        - 2 * worker_gradients @ worker_gradients.transpose(0, 1)
    ).clamp_min(0)
    squared_distances.fill_diagonal_(float("inf"))

    neighbor_count = num_workers - num_byzantine - 2
    closest_distances = torch.topk(
        squared_distances,
        k=neighbor_count,
        dim=1,
        largest=False,
    ).values
    return closest_distances.sum(dim=1)


def krum_aggregate(
    worker_gradients: Tensor,
    num_byzantine: int,
) -> AggregationResult:
    """Select the single lowest-scoring worker gradient using Krum."""
    scores = _krum_scores(worker_gradients, num_byzantine)
    selected_worker = scores.argmin()

    return AggregationResult(
        gradient=worker_gradients[selected_worker].clone(),
        selected_worker_ids=selected_worker.reshape(1),
        scores=scores,
    )


def multi_krum_aggregate(
    worker_gradients: Tensor,
    num_byzantine: int,
    num_selected: int,
) -> AggregationResult:
    """Average the ``num_selected`` lowest-scoring gradients using Multi-Krum."""
    num_workers = worker_gradients.size(0) if worker_gradients.ndim >= 1 else 0
    if num_selected <= 0 or num_selected > num_workers:
        raise ValueError(
            "Multi-Krum num_selected must be in [1, num_workers], "
            f"got num_selected={num_selected}, num_workers={num_workers}"
        )

    scores = _krum_scores(worker_gradients, num_byzantine)
    # Stable sorting makes equal-score ties deterministic by preserving the
    # original worker-ID order.
    selected_workers = torch.argsort(scores, stable=True)[:num_selected]

    return AggregationResult(
        gradient=worker_gradients[selected_workers].mean(dim=0),
        selected_worker_ids=selected_workers,
        scores=scores,
    )


def aggregate_gradients(
    worker_gradients: Tensor,
    aggregator: str,
    krum_f: int = 0,
    multi_krum_m: int | None = None,
) -> AggregationResult:
    """Dispatch worker gradients to the selected aggregation rule."""
    if aggregator == "mean":
        gradient = mean_aggregate(worker_gradients)
        selected_worker_ids = torch.arange(
            worker_gradients.size(0), device=worker_gradients.device
        )
        return AggregationResult(
            gradient=gradient,
            selected_worker_ids=selected_worker_ids,
        )
    if aggregator == "krum":
        return krum_aggregate(worker_gradients, num_byzantine=krum_f)
    if aggregator == "multi-krum":
        if multi_krum_m is None:
            raise ValueError("Multi-Krum requires multi_krum_m")
        return multi_krum_aggregate(
            worker_gradients,
            num_byzantine=krum_f,
            num_selected=multi_krum_m,
        )
    raise ValueError(f"Unknown aggregator: {aggregator}")


def assign_flat_gradient(model: nn.Module, flat_gradient: Tensor) -> None:
    """Split a flat gradient and assign each slice to its model parameter."""
    expected_size = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if flat_gradient.ndim != 1 or flat_gradient.numel() != expected_size:
        raise ValueError(
            f"Expected a flat gradient with {expected_size} values, "
            f"got shape {tuple(flat_gradient.shape)}"
        )

    offset = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        next_offset = offset + parameter.numel()
        parameter.grad = flat_gradient[offset:next_offset].view_as(parameter).clone()
        offset = next_offset
