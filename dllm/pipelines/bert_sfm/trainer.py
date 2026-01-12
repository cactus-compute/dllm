"""
Fisher-Rao Flow Matching Trainer for BERT.

This trainer implements endpoint prediction + cross-entropy training on the
Fisher-Rao manifold (positive orthant of the unit hypersphere).

References:
- CE_TRAINING.md for training setup details
- Fisher-Rao geometry for categorical distributions
"""

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.configs import TrainingArguments
from dllm.core.trainers.utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback
from dllm.pipelines.bert_sfm.sampler import BertSFMSampler, BertSFMSamplerConfig
from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map,
    log_map,
    parallel_transport,
    make_tangent,
    simplex_to_sphere,
    sphere_to_simplex,
    uniform_prior,
    linear_schedule,
    cosine_schedule,
    mse_velocity_loss_to_onehot,
)


def geodesic_interpolant_to_onehot(
    x_0: torch.Tensor, target_indices: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Optimized geodesic interpolation when x_1 is a one-hot vector on the sphere.

    When x_1 = e_k (one-hot at index k), the geodesic simplifies significantly.
    On the sphere, sqrt(one-hot) = one-hot, so x_1[k] = 1 and x_1[j] = 0 for j != k.

    The geodesic from x_0 to x_1 at time t is:
        x_t = x_0 * cos(t * theta) + (x_1 - x_0 * cos(theta)) * sin(t * theta) / sin(theta)

    where theta = arccos(x_0[k]) is the angle between x_0 and x_1.

    Args:
        x_0: Start point on sphere, shape (B, L, V)
        target_indices: Indices of the one-hot targets, shape (B, L)
        t: Interpolation parameter in [0, 1], shape (B,)

    Returns:
        Interpolated point on sphere, shape (B, L, V)
    """
    b, l, v = x_0.shape
    dtype = x_0.dtype

    # Expand t to (B, L, 1) for broadcasting, matching x_0's dtype
    t = t.to(dtype).view(b, 1, 1).expand(b, l, 1)

    # Get x_0's component at the target index: x_0[k] for each position
    # This is cos(theta) where theta is the geodesic distance
    x_0_at_target = x_0.gather(dim=-1, index=target_indices.unsqueeze(-1))  # (B, L, 1)
    cos_theta = x_0_at_target.clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta).to(dtype)  # (B, L, 1)

    # Compute sin values (cast back to dtype after trig ops)
    sin_theta = torch.sin(theta).to(dtype).clamp(min=1e-8)
    sin_t_theta = torch.sin(t * theta).to(dtype)
    cos_t_theta = torch.cos(t * theta).to(dtype)

    # x_t = x_0 * cos(t*theta) + direction * sin(t*theta)
    # where direction = (x_1 - x_0 * cos(theta)) / sin(theta)
    #
    # For non-target indices j != k:
    #   x_t[j] = x_0[j] * cos(t*theta) - x_0[j] * cos(theta) * sin(t*theta) / sin(theta)
    #          = x_0[j] * (cos(t*theta) - cos(theta) * sin(t*theta) / sin(theta))
    #
    # For target index k:
    #   x_t[k] = x_0[k] * cos(t*theta) + (1 - x_0[k] * cos(theta)) * sin(t*theta) / sin(theta)

    # Coefficient for all positions (works for j != k)
    coeff = cos_t_theta - cos_theta * sin_t_theta / sin_theta  # (B, L, 1)
    x_t = x_0 * coeff  # (B, L, V)

    # Correction for target index k: add (1 - x_0[k]*cos(theta)) * sin(t*theta) / sin(theta) - (existing contribution)
    # Existing contribution at k: x_0[k] * coeff
    # Correct value at k: x_0[k] * cos(t*theta) + (1 - x_0[k]*cos(theta)) * sin(t*theta) / sin(theta)
    target_val = cos_theta * cos_t_theta + (1 - cos_theta * cos_theta) * sin_t_theta / sin_theta

    # Scatter the correct value at target positions
    x_t = x_t.scatter(dim=-1, index=target_indices.unsqueeze(-1), src=target_val)

    return x_t


# ============== Trainer ==============


class BertSFMTrainer(transformers.Trainer):
    """
    Fisher-Rao Flow Matching Trainer for BERT-style models.

    This trainer implements continuous-time flow matching on the Fisher-Rao
    manifold. Instead of discrete masking, it:
    1. Maps one-hot targets to the positive orthant of the unit sphere
    2. Samples noise from a uniform prior on the same manifold
    3. Interpolates along geodesics between noise and data
    4. Trains the model to predict endpoint or velocity from the interpolant
    5. Uses cross-entropy (endpoint) or MSE (velocity) loss
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
        embed_type: str = "spherical"  # "spherical" or "simplex"
        loss_type: str = "ce"  # "ce" (cross-entropy) or "mse" (velocity MSE)
        eval_integration_steps: int = 20  # Number of integration steps for evaluation
        weight_decay: float = 0.01  # AdamW weight decay (L2 regularization)
        # Dataloader optimizations - defaults set based on CUDA availability
        dataloader_num_workers: int = 8 if torch.cuda.is_available() else 0
        dataloader_pin_memory: bool = torch.cuda.is_available()
        dataloader_prefetch_factor: int | None = 2 if torch.cuda.is_available() else None
        # Self-consistency training (exposes model to off-geodesic states during training)
        self_consistency_prob: float = 0.0  # 0.0 = disabled, 0.5 = 50% of batches
        self_consistency_max_steps: int = 5  # Maximum simulation steps (for "simulate" mode)
        self_consistency_schedule: str = "constant"  # "constant" or "linear_ramp"
        # Self-consistency mode: "simulate" (old) or "noise" (new, recommended)
        # "noise" adds tangent space noise to x_t instead of simulating integration
        # This is cheaper and works better early in training
        self_consistency_mode: str = "noise"
        # Noise scale for "noise" mode - controls perturbation magnitude
        # The actual noise is scaled by t (more noise at later timesteps where errors accumulate)
        self_consistency_noise_scale: float = 0.1
        # Step weight capping during evaluation to prevent blow-up near t=1
        # 0 = no cap, e.g. 4.0 = cap at 4x dt (helps when model predictions are imperfect)
        eval_step_weight_cap: float = 0.0
        # Geodesic loss: auxiliary loss that penalizes when sqrt(softmax(logits)) is far from
        # the true endpoint on the sphere. This aligns training with inference behavior.
        # 0 = disabled, try 0.1-1.0 for moderate regularization
        geodesic_loss_weight: float = 0.0
        # Hybrid loss: combine CE (endpoint) and MSE (velocity) losses
        # This is more principled than geodesic_loss - it adds the standard flow matching
        # velocity MSE loss as a regularizer to the CE loss.
        # 0 = disabled (pure CE), try 0.1-1.0 for hybrid training
        mse_loss_weight: float = 0.0

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
        if args.embed_type not in ("spherical", "simplex"):
            raise ValueError(f"embed_type must be 'spherical' or 'simplex', got '{args.embed_type}'")
        if args.loss_type not in ("ce", "mse"):
            raise ValueError(f"loss_type must be 'ce' or 'mse', got '{args.loss_type}'")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.schedule_type = args.schedule_type
        self.schedule_nu = args.schedule_nu
        self.time_weight_min = args.time_weight_min
        self.time_weight_max = args.time_weight_max
        self.embed_type = args.embed_type
        self.loss_type = args.loss_type
        self.eval_integration_steps = args.eval_integration_steps
        # Self-consistency training
        self.self_consistency_prob = args.self_consistency_prob
        self.self_consistency_max_steps = args.self_consistency_max_steps
        self.self_consistency_schedule = args.self_consistency_schedule
        self.self_consistency_mode = args.self_consistency_mode
        self.self_consistency_noise_scale = args.self_consistency_noise_scale
        # Step weight capping for evaluation
        self.eval_step_weight_cap = args.eval_step_weight_cap
        # Geodesic loss
        self.geodesic_loss_weight = args.geodesic_loss_weight
        # Hybrid MSE loss weight
        self.mse_loss_weight = args.mse_loss_weight

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
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss weights given timestep t.

        Args:
            t: Timestep values, shape (B,)
            alpha_t: Interpolation values, shape (B,)
            input_ids: Input token IDs, shape (B, L)

        Returns:
            Loss weights, shape (B, L)
        """
        b, l = input_ids.shape

        if self.loss_weight_type == "uniform":
            return torch.ones((b, l), device=input_ids.device)
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
    def _simulate_integration(
        self,
        x_0: torch.Tensor,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        model: nn.Module,
        embed_layer: nn.Module,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate integration steps to get off-geodesic states for self-consistency training.

        Instead of using the exact geodesic interpolant x_t, this simulates the actual
        inference process: starting from x_0, taking integration steps using the model
        itself to get an off-geodesic state that the model will encounter during inference.

        Key insight: We simulate from t=0 toward t=1 using a fixed number of steps
        (like inference), then return the state at the step closest to the target time t.
        This ensures the model sees realistic off-geodesic states that accumulate error
        over multiple integration steps.

        Args:
            x_0: Starting point on sphere (noise from prior), shape (B, L, V)
            input_ids: Ground truth token indices, shape (B, L)
            t: Target time to return state at, shape (B,)
            model: The language model
            embed_layer: Embedding layer for soft embeddings
            loss_mask: Boolean mask for flow positions (True = flow, False = prompt)

        Returns:
            x_t: Simulated state on sphere (off-geodesic), shape (B, L, V)
        """
        b, l, v = x_0.shape
        device = x_0.device
        compute_dtype = embed_layer.weight.dtype

        # Use fixed number of steps (like inference) to ensure consistent error accumulation
        num_steps = self.self_consistency_max_steps

        # Time grid from 0 to 1 (full integration like inference)
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        dt = 1.0 / num_steps

        # Find which step index corresponds to target time t (use mean for batch)
        t_mean = t.mean().item()
        target_step = min(int(t_mean * num_steps), num_steps - 1)
        target_step = max(1, target_step)  # At least 1 step

        # Start from x_0
        x_current = x_0.clone()

        # Prepare prompt one-hot on sphere for masking
        prompt_onehot = F.one_hot(input_ids, num_classes=v).to(compute_dtype)
        prompt_sphere = simplex_to_sphere(prompt_onehot)
        flow_mask_expanded = loss_mask.unsqueeze(-1).expand_as(x_current)

        # Keep prompt positions as clean one-hot
        x_current = torch.where(flow_mask_expanded, x_current, prompt_sphere)

        # Simulate integration steps until we reach the target step
        for step in range(target_step):
            t_curr = timesteps[step]

            # Get schedule values for step weight
            alpha_t, alpha_t_prime = self._get_schedule(t_curr.unsqueeze(0))

            # Compute soft embeddings from current sphere state
            x_embed = x_current if self.embed_type == "spherical" else sphere_to_simplex(x_current)
            soft_embeddings = torch.matmul(x_embed.to(compute_dtype), embed_layer.weight)

            # Forward pass
            outputs = model(inputs_embeds=soft_embeddings)
            logits = outputs.logits  # (B, L, V)

            # Convert logits to sphere prediction (endpoint prediction)
            probs = F.softmax(logits, dim=-1)
            x_1_pred = probs.sqrt()  # sphere representation

            # Compute step weight: alpha'(t) * dt / (1 - alpha(t))
            step_weight = alpha_t_prime * dt / (1 - alpha_t + 1e-5)

            # Take geodesic step toward predicted endpoint
            tangent = log_map(x_current, x_1_pred) * step_weight
            x_new = exp_map(x_current, tangent)

            # Project to sphere for numerical stability
            x_new = x_new / torch.norm(x_new, dim=-1, keepdim=True).clamp(min=1e-8)

            # Only update flow positions; keep prompt positions fixed
            x_current = torch.where(flow_mask_expanded, x_new, prompt_sphere)

        return x_current

    def _add_tangent_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        loss_mask: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        """
        Add noise in the tangent space at x_t to simulate off-geodesic states.

        This is a simpler and more effective alternative to full simulation.
        The noise magnitude scales with t (more noise at later times where
        integration errors accumulate more).

        Args:
            x_t: Current state on sphere, shape (B, L, V)
            t: Current time values, shape (B,)
            loss_mask: Boolean mask for flow positions (True = flow, False = prompt)
            noise_scale: Base scale for the noise

        Returns:
            x_t_noisy: Perturbed state on sphere, shape (B, L, V)
        """
        b, l, v = x_t.shape
        device = x_t.device
        dtype = x_t.dtype

        # Generate random noise in R^V
        noise = torch.randn(b, l, v, device=device, dtype=dtype)

        # Project noise to tangent space at x_t: noise - <x_t, noise> * x_t
        dot = (x_t * noise).sum(dim=-1, keepdim=True)
        tangent_noise = noise - dot * x_t

        # Scale noise by t (more noise at later times) and base scale
        # At t=0, no noise; at t=1, full noise_scale
        t_expanded = t.view(b, 1, 1).expand(b, l, 1)
        scaled_noise = tangent_noise * t_expanded * noise_scale

        # Move along tangent direction using exp_map
        x_t_noisy = exp_map(x_t, scaled_noise)

        # Project back to sphere for numerical stability
        x_t_noisy = x_t_noisy / torch.norm(x_t_noisy, dim=-1, keepdim=True).clamp(min=1e-8)

        # Only perturb flow positions; keep prompt positions clean
        flow_mask_expanded = loss_mask.unsqueeze(-1).expand_as(x_t)
        x_t_noisy = torch.where(flow_mask_expanded, x_t_noisy, x_t)

        return x_t_noisy

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Evaluation via full flow integration from t=0 to t=1 using the sampler.

        Uses BertSFMSampler.flow_integrate() to:
        1. Start from pure noise (t=0) for response positions, clean one-hot for prompt
        2. Integrate over 20 timesteps using Euler integration on the manifold
        3. Compute PPL on the final generated distribution vs ground truth

        Memory-optimized: avoids redundant [B, L, V] allocations by initializing
        x_sphere directly and using in-place updates where possible.
        """

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        unwrapped_model = model.module if hasattr(model, "module") else model
        vocab_size = unwrapped_model.config.vocab_size
        device = input_ids.device

        # Positions where we compute loss (response positions, not prompt)
        loss_mask = labels != -100  # [b, l]
        prompt_mask = ~loss_mask  # [b, l]

        # Get embedding layer
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        else:
            embed_layer = model.model.embed_tokens

        # Initialize sphere state: start with uniform prior for ALL positions
        # This avoids creating separate prompt_onehot and prior_sample tensors
        x_sphere = uniform_prior((b, l, vocab_size), device=device)

        # For prompt positions, overwrite with one-hot on sphere (in-place)
        # One-hot on sphere: all zeros except 1.0 at the token index
        if prompt_mask.any():
            x_sphere[prompt_mask] = 0
            prompt_indices = input_ids[prompt_mask].unsqueeze(-1)
            x_sphere[prompt_mask] = x_sphere[prompt_mask].scatter(-1, prompt_indices, 1.0)
            del prompt_indices
        del prompt_mask

        # Get discrete embeddings for context positions
        context_embeds = embed_layer(input_ids)

        # Use sampler to do flow integration
        sampler = BertSFMSampler(model=model, tokenizer=self.processing_class)
        config = BertSFMSamplerConfig(
            steps=self.eval_integration_steps,
            temperature=0.0,
            schedule_type=self.schedule_type,
            schedule_nu=self.schedule_nu,
            embed_type=self.embed_type,
            step_weight_cap=self.eval_step_weight_cap,
        )

        # Run flow integration (reuses the core loop)
        x_sphere, _ = sampler.flow_integrate(
            x_sphere=x_sphere,
            flow_mask=loss_mask,
            context_embeds=context_embeds,
            attention_mask=attention_mask,
            config=config,
            steps=self.eval_integration_steps,
            temperature=0.0,
            inference_scaling=1.0,
            return_histories=False,
        )
        del context_embeds

        # Convert sphere to log-probs directly: sphere_to_simplex squares, so log(x^2) = 2*log(x)
        # This avoids allocating a separate final_probs tensor
        # x_sphere^2 = probs, so 2*log(x_sphere) = log(probs)
        x_sphere.clamp_(min=1e-5)
        final_log_probs = x_sphere.log_().mul_(2)  # in-place: log then scale by 2

        # Use nll_loss since we already have log probs (not cross_entropy which applies log_softmax)
        token_nll = F.nll_loss(
            final_log_probs.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        del final_log_probs  # Free memory before metric update
        token_nll = token_nll * loss_mask.float()

        # Update metrics
        self.meter.update(
            split="eval",
            value=token_nll.detach(),
            weight=loss_mask.float().detach(),
        )

        # Normalize loss
        num_tokens = loss_mask.sum().clamp_min(1)
        del loss_mask
        loss = token_nll.sum() / num_tokens
        del token_nll

        # Always return prediction_loss_only=True style to avoid keeping final_logits
        return (loss.detach(), None, None)

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

        # Get the embedding layer and its dtype (set by accelerate/deepspeed mixed precision)
        # Use unwrapped_model to handle DDP wrapper
        if hasattr(unwrapped_model, "get_input_embeddings"):
            embed_layer = unwrapped_model.get_input_embeddings()
        else:
            embed_layer = unwrapped_model.model.embed_tokens
        compute_dtype = embed_layer.weight.dtype

        # === 1. Sample diffusion timesteps ===
        # t ∈ [ε, 1) to avoid degenerate values
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)

        # Get interpolation schedule
        alpha_t, alpha_t_prime = self._get_schedule(t)  # Both shape (B,)

        # === 2. Sample noise from uniform prior on sphere ===
        x_0 = uniform_prior((b, l, vocab_size), device=device, dtype=compute_dtype)  # [b, l, V]

        # === 3. Compute x_t via geodesic interpolation ===
        # Since x_1 is always one-hot (sqrt of one-hot = one-hot on sphere),
        # we use an optimized path that avoids materializing the full x_1 tensor
        x_t = geodesic_interpolant_to_onehot(x_0, input_ids, alpha_t)  # [b, l, V]

        # For MSE loss (or hybrid CE+MSE), we need x_0 to compute velocity target; otherwise free it
        if self.loss_type == "ce" and self.mse_loss_weight == 0:
            del x_0

        # === 3b. Keep prompt positions clean (not noised) ===
        # For positions where loss_mask=False (prompt), use the clean one-hot on sphere
        # This teaches the model to condition on clean prompts while denoising targets
        if not loss_mask.all():
            prompt_mask = ~loss_mask
            x_t[prompt_mask] = 0
            prompt_indices = input_ids[prompt_mask].unsqueeze(-1)
            x_t[prompt_mask] = x_t[prompt_mask].scatter(-1, prompt_indices, 1.0)

        # === 3c. Self-consistency: add perturbation to expose model to off-geodesic states ===
        # Determine if we should use self-consistency for this batch
        if self.self_consistency_schedule == "linear_ramp":
            training_progress = self.state.global_step / max(self.state.max_steps, 1)
            effective_sc_prob = min(self.self_consistency_prob, training_progress * self.self_consistency_prob * 2)
        else:
            effective_sc_prob = self.self_consistency_prob

        use_self_consistency = (
            effective_sc_prob > 0
            and random.random() < effective_sc_prob
            and model.training
        )

        if use_self_consistency:
            if self.self_consistency_mode == "noise":
                # Noise mode: add tangent space noise to x_t
                # This is cheap (no extra forward passes) and works from the start of training
                x_t = self._add_tangent_noise(x_t, t, loss_mask, self.self_consistency_noise_scale)
            elif self.self_consistency_mode == "simulate":
                # Simulate mode: run integration with the model to get off-geodesic states
                # This is more expensive but may be more realistic
                # Note: we need x_0 for simulation, so re-sample it
                x_0_sim = uniform_prior((b, l, vocab_size), device=device, dtype=compute_dtype)
                x_t = self._simulate_integration(x_0_sim, input_ids, alpha_t, model, embed_layer, loss_mask)
            else:
                raise ValueError(f"Unknown self_consistency_mode: {self.self_consistency_mode}")

        # === 5. Forward pass ===
        # Compute soft embeddings: x_embed @ embedding_matrix
        # x_t is on sphere; convert to simplex if embed_type == "simplex"
        x_embed = x_t if self.embed_type == "spherical" else sphere_to_simplex(x_t)
        # x_embed: [b, l, V], embed_weight: [V, D] -> [b, l, D]
        # Ensure dtype matches embedding layer (important for mixed precision training)
        soft_embeddings = torch.matmul(x_embed.to(compute_dtype), embed_layer.weight)

        # Forward pass with soft embeddings
        # Most HuggingFace models accept inputs_embeds
        outputs = model(
            inputs_embeds=soft_embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [b, l, V]

        # === 6. Compute per-token loss weights ===
        loss_weights = self._compute_loss_weights(t, alpha_t, input_ids)  # [b, l]

        # === 7. Compute loss based on loss_type ===
        if self.loss_type == "ce":
            # Cross-entropy loss: target is the original tokens (endpoint prediction)
            token_loss = F.cross_entropy(
                logits.transpose(1, 2),  # [b, V, l]
                input_ids,  # [b, l]
                reduction="none",  # [b, l]
            )

            # === 7b. Optional: Geodesic loss to align training with inference ===
            # During inference, we use sqrt(softmax(logits)) as the predicted endpoint.
            # CE loss only ensures the correct token has high probability, not that
            # sqrt(softmax) lies on the geodesic toward the correct endpoint.
            # This auxiliary loss directly penalizes geodesic distance.
            if self.geodesic_loss_weight > 0:
                # Predicted endpoint on sphere: sqrt(softmax(logits))
                probs = F.softmax(logits, dim=-1)  # [b, l, V]
                x_pred = probs.sqrt()  # [b, l, V] - predicted endpoint on sphere

                # True endpoint on sphere: one-hot (which equals its sqrt)
                # x_true[i, j, k] = 1 if k == input_ids[i, j] else 0
                # Compute geodesic distance: arccos(<x_pred, x_true>)
                # <x_pred, x_true> = x_pred[..., input_ids] = the predicted sqrt-prob at true token
                x_pred_at_target = x_pred.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [b, l]

                # Geodesic distance = arccos(dot product), but dot product = x_pred_at_target
                # since x_true is one-hot (all zeros except 1 at target position)
                geodesic_dist = torch.acos(x_pred_at_target.clamp(-1 + 1e-6, 1 - 1e-6))  # [b, l]

                # Weight by (1-t) to emphasize errors near t=1 where integration is more critical
                # At t close to 1, we're near the endpoint and errors matter most
                t_weight = (1 - alpha_t).view(b, 1).expand(b, l)
                geodesic_loss_per_token = geodesic_dist * t_weight  # [b, l]

                # Add to token_loss with weighting
                token_loss = token_loss + self.geodesic_loss_weight * geodesic_loss_per_token

            # === 7c. Optional: Hybrid MSE loss for velocity regularization ===
            # This adds the standard flow matching MSE velocity loss as a regularizer.
            # The CE loss teaches the model to predict the correct endpoint (token),
            # while the MSE loss teaches proper velocity/geometry on the sphere.
            # Uses memory-efficient computation that exploits one-hot structure of x_1.
            if self.mse_loss_weight > 0:
                # Project model output (logits) to tangent space at x_t
                predicted_velocity = make_tangent(x_t, logits)  # [b, l, V]

                # Use memory-efficient MSE that doesn't materialize full (b, l, V) tensors
                # for x_1, log_map, or parallel transport
                mse_loss_per_token = mse_velocity_loss_to_onehot(
                    x_0, x_t, input_ids, predicted_velocity
                )  # [b, l]

                # Add to token_loss with weighting
                token_loss = token_loss + self.mse_loss_weight * mse_loss_per_token

        elif self.loss_type == "mse":
            # Velocity MSE loss: target is the velocity (tangent vector)
            # Construct x_1 (one-hot on sphere) for target positions
            x_1 = F.one_hot(input_ids, num_classes=vocab_size).to(compute_dtype)  # [b, l, V]
            # sqrt(one-hot) = one-hot on sphere

            # Compute target velocity: log_map(x_0, x_1) parallel transported to x_t
            # velocity at x_0 pointing toward x_1
            velocity_at_x0 = log_map(x_0, x_1)  # [b, l, V]
            # Parallel transport to x_t
            target_velocity = parallel_transport(x_0, x_t, velocity_at_x0)  # [b, l, V]
            del x_0, x_1, velocity_at_x0  # Free memory

            # Project model output (logits) to tangent space at x_t
            predicted_velocity = make_tangent(x_t, logits)  # [b, l, V]

            # Compute MSE loss per token: sum over vocab dimension
            token_loss = (predicted_velocity - target_velocity).square().sum(dim=-1)  # [b, l]
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

        # Update metrics with raw loss (before weighting) for accurate tracking
        self.meter.update(
            split="train" if model.training else "eval",
            value=(token_loss * loss_mask.float()).detach(),
            weight=loss_mask.float().detach(),
        )

        # Apply loss weights and mask for backprop
        token_loss = token_loss * loss_weights * loss_mask.float()  # [b, l]

        # === 8. Normalize loss ===
        if self.loss_norm_type == "token":
            token_loss = token_loss / loss_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_loss = token_loss / (loss_mask.sum(-1, keepdim=True).clamp_min(1) * b)
        elif self.loss_norm_type == "batch":
            token_loss = token_loss / b
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")

        loss = token_loss.sum()

        return (loss, outputs) if return_outputs else loss
