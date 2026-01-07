"""
Fisher-Rao Flow Matching Trainer for BERT.

This trainer implements endpoint prediction + cross-entropy training on the
Fisher-Rao manifold (positive orthant of the unit hypersphere).

References:
- CE_TRAINING.md for training setup details
- Fisher-Rao geometry for categorical distributions
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.configs import TrainingArguments
from dllm.core.trainers.utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback


# ============== Manifold Operations ==============


def simplex_to_sphere(p: torch.Tensor) -> torch.Tensor:
    """Map probability vector to sphere via square root."""
    return torch.sqrt(p.clamp(min=1e-8))


def sphere_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """Map sphere point back to simplex via squaring."""
    return x**2


def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on the sphere.

    Move from point p in direction v (tangent vector) on the sphere.

    Args:
        p: Point on the sphere, shape (..., D)
        v: Tangent vector at p, shape (..., D)

    Returns:
        New point on the sphere, shape (..., D)
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    return p * torch.cos(v_norm) + v * torch.sin(v_norm) / v_norm


def log_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map on the sphere.

    Compute the tangent vector at p pointing toward q.

    Args:
        p: Source point on the sphere, shape (..., D)
        q: Target point on the sphere, shape (..., D)

    Returns:
        Tangent vector at p, shape (..., D)
    """
    # Project q onto tangent space at p
    dot_pq = (p * q).sum(dim=-1, keepdim=True)
    q_proj = q - dot_pq * p
    q_proj_norm = torch.norm(q_proj, dim=-1, keepdim=True).clamp(min=1e-8)

    # Compute geodesic distance
    dot = dot_pq.clamp(-1 + 1e-7, 1 - 1e-7)
    dist = torch.acos(dot)

    return q_proj / q_proj_norm * dist


def geodesic_interpolant(
    x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Interpolate along geodesic from x_0 to x_1.

    Args:
        x_0: Start point on sphere, shape (B, L, V) or (B, V)
        x_1: End point on sphere, shape (B, L, V) or (B, V)
        t: Interpolation parameter in [0, 1], shape (B,) or (B, 1) or (B, 1, 1)

    Returns:
        Interpolated point on sphere, same shape as x_0
    """
    # Expand t to match x_0 dimensions
    while t.dim() < x_0.dim():
        t = t.unsqueeze(-1)
    return exp_map(x_0, t * log_map(x_0, x_1))


def uniform_prior(shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Sample uniformly from positive orthant of the sphere.

    Args:
        shape: Shape of output tensor, last dim is the manifold dimension
        device: Device to create tensor on

    Returns:
        Points on positive orthant of unit sphere
    """
    # Sample from standard normal and take absolute value for positive orthant
    x = torch.randn(shape, device=device).abs()
    # Project to sphere
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)


def project_to_sphere(x: torch.Tensor) -> torch.Tensor:
    """Project to unit sphere for numerical stability."""
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)


# ============== Interpolation Schedules ==============


def linear_schedule(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Linear interpolation schedule.

    Returns:
        (alpha_t, alpha_t_prime): Interpolation value and its derivative
    """
    return t, torch.ones_like(t)


def cosine_schedule(
    t: torch.Tensor, nu: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine interpolation schedule.

    Args:
        t: Time values in [0, 1]
        nu: Schedule parameter (default 1.0)

    Returns:
        (alpha_t, alpha_t_prime): Interpolation value and its derivative
    """
    import math

    t_pow_nu = t.pow(nu)
    alpha_t = 1 - torch.cos(math.pi / 2 * t_pow_nu).square()
    alpha_t_prime = (
        math.pi / 2 * torch.sin(math.pi * t_pow_nu) * nu * t.pow(nu - 1).clamp(min=1e-8)
    )
    return alpha_t, alpha_t_prime


# ============== Trainer ==============


class BertSFMTrainer(transformers.Trainer):
    """
    Fisher-Rao Flow Matching Trainer for BERT-style models.

    This trainer implements continuous-time flow matching on the Fisher-Rao
    manifold. Instead of discrete masking, it:
    1. Maps one-hot targets to the positive orthant of the unit sphere
    2. Samples noise from a uniform prior on the same manifold
    3. Interpolates along geodesics between noise and data
    4. Trains the model to predict the endpoint (data) from the interpolant
    5. Uses cross-entropy loss against the true tokens
    """

    @dataclass
    class BertSFMConfig(TrainingArguments):
        time_epsilon: float = 1e-3
        loss_weight_type: str = "uniform"  # "time_weighted", "uniform"
        loss_norm_type: str = "token"  # "batch", "sequence", "token"
        schedule_type: str = "linear"  # "linear", "cosine"
        schedule_nu: float = 1.0  # Parameter for cosine schedule
        time_weight_min: float = 0.05  # Min clamp for time weighting
        time_weight_max: float = 1.5  # Max clamp for time weighting

    def __init__(
        self,
        args: BertSFMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.schedule_type = args.schedule_type
        self.schedule_nu = args.schedule_nu
        self.time_weight_min = args.time_weight_min
        self.time_weight_max = args.time_weight_max

        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)

    def _get_schedule(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get interpolation schedule values."""
        if self.schedule_type == "linear":
            return linear_schedule(t)
        elif self.schedule_type == "cosine":
            return cosine_schedule(t, nu=self.schedule_nu)
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute loss weights given timestep t.

        Args:
            t: Timestep values, shape (B,)
            alpha_t: Interpolation values, shape (B,)
            inputs: Input dictionary with input_ids

        Returns:
            Loss weights, shape (B, L)
        """
        b, l = inputs["input_ids"].shape

        if self.loss_weight_type == "uniform":
            return torch.ones((b, l), device=inputs["input_ids"].device)
        elif self.loss_weight_type == "time_weighted":
            # Weight by alpha_t / (1 - alpha_t) - emphasizes samples near t=1
            weights = alpha_t / (1 - alpha_t + 1e-5)
            weights = torch.clamp(weights, min=self.time_weight_min, max=self.time_weight_max)
            return weights.unsqueeze(1).expand(b, l)
        elif self.loss_weight_type == "scheduler":
            # Use the alpha scheduler's weight function
            loss_weights = self.scheduler.weight(t).unsqueeze(1).expand(b, l)
            return loss_weights
        else:
            raise ValueError(f"Unknown loss_weight_type: {self.loss_weight_type}")

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the Fisher-Rao flow matching loss.

        Instead of discrete masking, this method:
        1. Converts input tokens to one-hot vectors on the probability simplex
        2. Maps them to the Fisher-Rao manifold (positive orthant of sphere)
        3. Samples uniform noise on the same manifold
        4. Interpolates along geodesics based on sampled timestep
        5. Model predicts logits for the original tokens
        6. Computes cross-entropy loss

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        assert self.processing_class.padding_side == "right"

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        # Handle DataParallel/DistributedDataParallel wrapped models
        unwrapped_model = model.module if hasattr(model, "module") else model
        vocab_size = unwrapped_model.config.vocab_size
        device = input_ids.device

        # Positions where we compute loss (not -100)
        loss_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        # t ∈ [ε, 1) to avoid degenerate values
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)

        # Get interpolation schedule
        alpha_t, alpha_t_prime = self._get_schedule(t)  # Both shape (B,)

        # === 2. Convert tokens to manifold representation ===
        # Create one-hot encoding of input tokens
        x_1_onehot = F.one_hot(input_ids, num_classes=vocab_size).float()  # [b, l, V]

        # Map one-hot (simplex) to sphere
        x_1 = simplex_to_sphere(x_1_onehot)  # [b, l, V]

        # === 3. Sample noise from uniform prior on sphere ===
        x_0 = uniform_prior((b, l, vocab_size), device=device)  # [b, l, V]

        # === 4. Geodesic interpolation ===
        # alpha_t needs shape (B, 1, 1) for broadcasting with (B, L, V)
        x_t = geodesic_interpolant(x_0, x_1, alpha_t)  # [b, l, V]

        # === 5. Forward pass ===
        # The model receives the interpolated points (soft embeddings)
        # We need to convert x_t back to a form the model can process
        # Since BERT expects discrete tokens, we use x_t as soft input embeddings

        # Get the embedding layer
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        else:
            embed_layer = model.model.embed_tokens

        # Compute soft embeddings: x_t @ embedding_matrix
        # x_t: [b, l, V], embed_weight: [V, D] -> [b, l, D]
        soft_embeddings = torch.matmul(x_t, embed_layer.weight)

        # Forward pass with soft embeddings
        # Most HuggingFace models accept inputs_embeds
        outputs = model(
            inputs_embeds=soft_embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [b, l, V]

        # === 6. Compute per-token loss weights ===
        loss_weights = self._compute_loss_weights(t, alpha_t, inputs)  # [b, l]

        # === 7. Compute cross-entropy loss ===
        # Target is the original tokens (endpoint prediction)
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )

        # Apply loss weights and mask
        token_nll = token_nll * loss_weights * loss_mask.float()  # [b, l]

        # Update metrics
        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=loss_mask.float().detach(),
        )

        # === 8. Normalize loss ===
        if self.loss_norm_type == "token":
            token_nll = token_nll / loss_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / (loss_mask.sum(-1, keepdim=True).clamp_min(1) * b)
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")

        loss = token_nll.sum()

        return (loss, outputs) if return_outputs else loss
