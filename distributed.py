"""Gradient operations for serial distributed SGD simulation."""

import torch
from torch import Tensor, nn


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
