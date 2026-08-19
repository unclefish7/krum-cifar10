"""Byzantine gradient attacks for the serial worker simulation."""

from collections.abc import Sequence

import torch
from torch import Tensor


def select_byzantine_workers(
    num_workers: int,
    num_byzantine: int,
    generator: torch.Generator,
) -> tuple[int, ...]:
    """Randomly select distinct Byzantine worker IDs for one round."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if num_byzantine <= 0 or num_byzantine >= num_workers:
        raise ValueError("num_byzantine must be in [1, num_workers)")
    permutation = torch.randperm(num_workers, generator=generator)
    return tuple(permutation[:num_byzantine].tolist())


def gaussian_attack(
    worker_gradients: Tensor,
    byzantine_worker_ids: Sequence[int],
    std: float = 200.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Replace Byzantine rows with zero-mean isotropic Gaussian vectors."""
    if worker_gradients.ndim != 2 or worker_gradients.size(0) == 0:
        raise ValueError("worker_gradients must have shape [num_workers, num_parameters]")
    if std < 0:
        raise ValueError("Gaussian attack std cannot be negative")

    worker_ids = list(byzantine_worker_ids)
    if not worker_ids:
        raise ValueError("Gaussian attack requires at least one Byzantine worker")
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("Byzantine worker IDs must be unique")
    if any(
        worker_id < 0 or worker_id >= worker_gradients.size(0)
        for worker_id in worker_ids
    ):
        raise ValueError("Byzantine worker ID is out of range")

    submitted_gradients = worker_gradients.clone()
    worker_index = torch.tensor(
        worker_ids,
        device=worker_gradients.device,
        dtype=torch.long,
    )
    gaussian_vectors = torch.randn(
        (len(worker_ids), worker_gradients.size(1)),
        device=worker_gradients.device,
        dtype=worker_gradients.dtype,
        generator=generator,
    ) * std
    submitted_gradients.index_copy_(0, worker_index, gaussian_vectors)
    return submitted_gradients


def apply_attack(
    worker_gradients: Tensor,
    attack: str,
    byzantine_worker_ids: Sequence[int],
    gaussian_std: float = 200.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return the gradients actually submitted to the aggregation rule."""
    if attack == "none":
        if byzantine_worker_ids:
            raise ValueError("attack='none' cannot have Byzantine worker IDs")
        return worker_gradients
    if attack == "gaussian":
        return gaussian_attack(
            worker_gradients,
            byzantine_worker_ids=byzantine_worker_ids,
            std=gaussian_std,
            generator=generator,
        )
    raise ValueError(f"Unknown attack: {attack}")
