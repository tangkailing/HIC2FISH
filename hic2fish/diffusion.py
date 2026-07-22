"""Cosine diffusion schedule and deterministic DDIM generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .constants import MATRIX_SIZE


def cosine_alpha_bar(
    diffusion_time: torch.Tensor,
    offset: float = 0.008,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Return the cumulative signal-retention coefficient."""
    alpha_bar = torch.cos(
        ((diffusion_time + offset) / (1.0 + offset)) * torch.pi / 2.0
    ) ** 2
    return alpha_bar.clamp(min=epsilon, max=1.0 - epsilon)


def initial_noise_for_seeds(
    seeds: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create one reproducible Gaussian initial state for every seed."""
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    samples: list[torch.Tensor] = []
    for seed in seeds:
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        samples.append(
            torch.randn(
                (1, MATRIX_SIZE, MATRIX_SIZE),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    return torch.stack(samples, dim=0)


@torch.inference_mode()
def deterministic_ddim_batch(
    model: nn.Module,
    condition: torch.Tensor,
    seeds: list[int],
    steps: int,
) -> torch.Tensor:
    """Generate normalized matrices with the final public-demo sampler.

    The reverse process uses eta=0. At every step, x0 is restricted to the
    normalized training range [0, 1], and epsilon is recomputed consistently
    from the restricted x0 estimate.
    """
    batch_size = len(seeds)
    if batch_size < 1:
        raise ValueError("At least one seed is required.")
    if steps < 1:
        raise ValueError("The number of DDIM steps must be positive.")

    device = condition.device
    current = initial_noise_for_seeds(seeds, device, condition.dtype)
    batch_condition = condition.expand(batch_size, -1, -1, -1)

    # Avoid evaluating the reverse process at the singular endpoint t=1.
    start_time = 1.0 - 1.0 / steps
    timesteps = torch.linspace(
        start_time,
        0.0,
        steps + 1,
        device=device,
        dtype=condition.dtype,
    )
    final_x0: torch.Tensor | None = None

    for step_index in range(steps):
        time = timesteps[step_index].expand(batch_size, 1)
        next_time = timesteps[step_index + 1].expand(batch_size, 1)
        predicted_noise = model(current, batch_condition, time)

        alpha_bar = cosine_alpha_bar(time).view(batch_size, 1, 1, 1)
        next_alpha_bar = cosine_alpha_bar(next_time).view(
            batch_size, 1, 1, 1
        )
        x0_estimate = (
            current - torch.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha_bar)

        x0_estimate = torch.clamp(x0_estimate, min=0.0, max=1.0)
        predicted_noise = (
            current - torch.sqrt(alpha_bar) * x0_estimate
        ) / torch.sqrt(1.0 - alpha_bar)

        current = (
            torch.sqrt(next_alpha_bar) * x0_estimate
            + torch.sqrt(1.0 - next_alpha_bar) * predicted_noise
        )
        final_x0 = x0_estimate

    if final_x0 is None:
        raise RuntimeError("DDIM did not produce a final x0 estimate.")
    lower_triangle = torch.tril(final_x0, diagonal=-1)
    return lower_triangle + lower_triangle.transpose(-1, -2)


def inverse_scale_distances(
    normalized: torch.Tensor,
    minimum_um: float,
    maximum_um: float,
) -> np.ndarray:
    """Restore normalized matrices to micrometres and enforce symmetry."""
    matrices = normalized[:, 0].detach().cpu().numpy().astype(np.float64)
    matrices = matrices * (maximum_um - minimum_um) + minimum_um
    lower_triangle = np.tril(matrices, k=-1)
    matrices = lower_triangle + np.transpose(lower_triangle, (0, 2, 1))
    diagonal = np.arange(MATRIX_SIZE)
    matrices[:, diagonal, diagonal] = 0.0
    return matrices


def generate_ensemble(
    model: nn.Module,
    condition: torch.Tensor,
    seeds: list[int],
    steps: int,
    batch_size: int,
    minimum_um: float,
    maximum_um: float,
) -> np.ndarray:
    """Generate a complete single-cell ensemble in bounded batches."""
    if batch_size < 1:
        raise ValueError("Generation batch size must be positive.")
    outputs: list[np.ndarray] = []
    total_batches = int(np.ceil(len(seeds) / batch_size))
    for batch_number, start in enumerate(
        range(0, len(seeds), batch_size), start=1
    ):
        batch_seeds = seeds[start : start + batch_size]
        normalized = deterministic_ddim_batch(
            model=model,
            condition=condition,
            seeds=batch_seeds,
            steps=steps,
        )
        outputs.append(
            inverse_scale_distances(normalized, minimum_um, maximum_um)
        )
        print(
            f"Generation batch {batch_number}/{total_batches} "
            f"(seeds {batch_seeds[0]}-{batch_seeds[-1]})"
        )
    return np.concatenate(outputs, axis=0)
