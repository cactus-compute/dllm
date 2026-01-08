"""
Fisher-Rao Flow Matching Sampler for BERT.

This sampler implements generation via endpoint-prediction flow matching on the
Fisher-Rao manifold (positive orthant of the unit hypersphere).

References:
- CE_SAMPLING.md for sampling setup details
- Fisher-Rao geometry for categorical distributions
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput

# ============== Manifold Operations ==============


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
    dot_pq = (p * q).sum(dim=-1, keepdim=True)
    q_proj = q - dot_pq * p
    q_proj_norm = torch.norm(q_proj, dim=-1, keepdim=True).clamp(min=1e-8)
    dot_clamped = dot_pq.clamp(-1 + 1e-7, 1 - 1e-7)
    dist = torch.acos(dot_clamped)
    return q_proj / q_proj_norm * dist


def project_to_sphere(x: torch.Tensor) -> torch.Tensor:
    """Project to unit sphere for numerical stability."""
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)


def simplex_to_sphere(p: torch.Tensor) -> torch.Tensor:
    """Map probability simplex to sphere via square root."""
    return torch.sqrt(p.clamp(min=1e-8))


def sphere_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """Map sphere point back to simplex via squaring."""
    return x**2


def uniform_prior(shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Sample uniformly from positive orthant of the sphere.

    Args:
        shape: Shape of output tensor, last dim is the manifold dimension
        device: Device to create tensor on

    Returns:
        Points on positive orthant of unit sphere
    """
    x = torch.randn(shape, device=device).abs()
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

    t_safe = t.clamp(min=1e-9)
    alpha_t = 1 - torch.cos(math.pi / 2 * t_safe.pow(nu)).square()
    alpha_t_prime = (
        math.pi / 2 * torch.sin(math.pi * t_safe.pow(nu)) * nu * t_safe.pow(nu - 1)
    )
    return alpha_t, alpha_t_prime


# ============== Sampler Config ==============


@dataclass
class BertSFMSamplerConfig(SamplerConfig):
    """Configuration for Fisher-Rao Flow Matching sampler."""

    max_new_tokens: int = 128
    steps: int = 20  # 20 timesteps for SFT evaluation
    temperature: float = 0.0
    schedule_type: str = "linear"  # "linear" or "cosine"
    schedule_nu: float = 1.0  # Parameter for cosine schedule
    inference_scaling: float = 1.0  # Scaling factor for step sizes


# ============== Sampler ==============


@dataclass
class BertSFMSampler(BaseSampler):
    """
    Fisher-Rao Flow Matching Sampler for BERT-style models.

    This sampler generates text by integrating the learned flow on the
    Fisher-Rao manifold from t=0 (prior) to t=1 (data).

    The process:
    1. Start from uniform prior on the positive orthant of the sphere
    2. At each step, predict the endpoint (target tokens) from current state
    3. Move along geodesic toward predicted endpoint
    4. Convert final manifold point to discrete tokens via argmax
    """

    def _get_schedule(
        self, t: torch.Tensor, config: BertSFMSamplerConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get interpolation schedule values."""
        if config.schedule_type == "linear":
            return linear_schedule(t)
        elif config.schedule_type == "cosine":
            return cosine_schedule(t, nu=config.schedule_nu)
        else:
            raise ValueError(f"Unknown schedule_type: {config.schedule_type}")

    @torch.no_grad()
    def flow_integrate(
        self,
        x_sphere: torch.Tensor,
        flow_mask: torch.Tensor,
        context_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        config: BertSFMSamplerConfig,
        steps: int,
        temperature: float,
        inference_scaling: float,
        return_histories: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Core flow integration loop on the Fisher-Rao manifold.

        Integrates from t=0 to t=1, updating only positions where flow_mask=True.

        Args:
            x_sphere: Initial sphere state, shape (B, T, V). Positions where
                flow_mask=True should be initialized from uniform prior.
            flow_mask: Boolean mask indicating which positions to flow (B, T).
                True = flow from noise, False = keep fixed.
            context_embeds: Discrete embeddings for context positions (B, T, D).
            attention_mask: Attention mask for the model (B, T).
            config: Sampler configuration.
            steps: Number of integration steps.
            temperature: Temperature for logits (0 = greedy).
            inference_scaling: Scaling factor for step sizes.
            return_histories: Whether to record token history at each step.

        Returns:
            Tuple of (final_x_sphere, histories) where histories is None if
            return_histories=False.
        """
        # Get embedding layer
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.model.embed_tokens

        # Time grid from 0 to 1
        device = x_sphere.device
        timesteps = torch.linspace(0, 1, steps + 1, device=device)

        histories = []
        if return_histories:
            init_tokens = sphere_to_simplex(x_sphere).argmax(dim=-1)
            histories.append(init_tokens.clone())

        # Integration loop
        for step_idx in range(steps):
            t_curr = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            dt = t_next - t_curr

            # Get schedule values
            alpha_t, alpha_t_prime = self._get_schedule(t_curr.unsqueeze(0), config)

            # Compute soft embeddings from current sphere state
            soft_embeddings = torch.matmul(
                x_sphere.to(embed_layer.weight.dtype), embed_layer.weight
            )

            # Use discrete embeddings for context, soft embeddings for flow positions
            combined_embeddings = torch.where(
                flow_mask.unsqueeze(-1).expand_as(soft_embeddings),
                soft_embeddings,
                context_embeds,
            )

            # Model forward pass
            outputs = self.model(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, T, V)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Convert logits to probabilities and then to sphere
            probs = F.softmax(logits, dim=-1)
            x_1_pred = simplex_to_sphere(probs)

            # Compute step weight: alpha'(t) * dt / (1 - alpha(t))
            step_weight = alpha_t_prime * dt / (1 - alpha_t + 1e-5)
            step_weight = step_weight * inference_scaling
            step_weight = step_weight.view(1, 1, 1)  # For broadcasting

            # Step along geodesic toward predicted endpoint
            tangent = log_map(x_sphere, x_1_pred)
            x_sphere_new = exp_map(x_sphere, tangent * step_weight)
            x_sphere_new = project_to_sphere(x_sphere_new)

            # Only update flow positions; keep context positions fixed
            x_sphere = torch.where(
                flow_mask.unsqueeze(-1).expand_as(x_sphere),
                x_sphere_new,
                x_sphere,
            )

            if return_histories:
                current_tokens = sphere_to_simplex(x_sphere).argmax(dim=-1)
                histories.append(current_tokens.clone())

        return x_sphere, histories if return_histories else None

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: BertSFMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Generate text using Fisher-Rao flow matching.

        Integrates the learned flow from t=0 to t=1, starting from the prior
        and moving toward the predicted data distribution at each step.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = BertSFMSamplerConfig()

        # Pull args from config, allow kwargs to override
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        temperature = kwargs.get("temperature", config.temperature)
        inference_scaling = kwargs.get("inference_scaling", config.inference_scaling)
        return_dict = kwargs.get("return_dict", config.return_dict)

        # Handle DataParallel/DistributedDataParallel wrapped models
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        device = unwrapped_model.device if hasattr(unwrapped_model, "device") else next(unwrapped_model.parameters()).device
        vocab_size = unwrapped_model.config.vocab_size
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # Convert inputs to tensors
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs
            ]

        B = len(inputs)
        prompt_lens = [p.shape[0] for p in inputs]
        T = max_new_tokens + max(prompt_lens)

        # Initialize canvas with prompts and generation positions
        x_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        for i, p in enumerate(inputs):
            x_ids[i, : prompt_lens[i]] = p

        # Build attention mask
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        for i, pl in enumerate(prompt_lens):
            valid_end = pl + max_new_tokens
            attention_mask[i, :valid_end] = 1

        # Create mask indicating which positions to generate (after prompt)
        gen_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for i, pl in enumerate(prompt_lens):
            gen_mask[i, pl : pl + max_new_tokens] = True

        # Get embedding layer
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.model.embed_tokens

        # Initialize sphere state
        # For prompt positions: use one-hot encoding mapped to sphere
        prompt_onehot = F.one_hot(x_ids, num_classes=vocab_size).float()
        x_sphere = simplex_to_sphere(prompt_onehot)

        # For generation positions: sample from uniform prior
        gen_prior = uniform_prior((B, max_new_tokens, vocab_size), device=device)
        for i, pl in enumerate(prompt_lens):
            x_sphere[i, pl : pl + max_new_tokens] = gen_prior[i, :max_new_tokens]

        # Get discrete prompt embeddings (fixed throughout generation)
        context_embeds = embed_layer(x_ids)  # (B, T, D)

        # Run flow integration
        x_sphere, histories = self.flow_integrate(
            x_sphere=x_sphere,
            flow_mask=gen_mask,
            context_embeds=context_embeds,
            attention_mask=attention_mask,
            config=config,
            steps=steps,
            temperature=temperature,
            inference_scaling=inference_scaling,
            return_histories=return_dict,
        )

        # Convert final sphere points to tokens
        final_probs = sphere_to_simplex(x_sphere)
        final_tokens = final_probs.argmax(dim=-1)

        # Merge prompt tokens with generated tokens
        output_ids = x_ids.clone()
        output_ids[gen_mask] = final_tokens[gen_mask]

        if return_dict:
            return SamplerOutput(sequences=output_ids, histories=histories)
        return output_ids

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: BertSFMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Fill in masked positions within input sequences.

        Uses flow matching to generate tokens at positions marked with mask_token.

        Args:
            inputs: List of sequences containing mask tokens to fill.
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with filled sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = BertSFMSamplerConfig()

        # Pull args from config, allow kwargs to override
        steps = kwargs.get("steps", config.steps)
        temperature = kwargs.get("temperature", config.temperature)
        inference_scaling = kwargs.get("inference_scaling", config.inference_scaling)
        return_dict = kwargs.get("return_dict", config.return_dict)

        # Handle DataParallel/DistributedDataParallel wrapped models
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        device = unwrapped_model.device if hasattr(unwrapped_model, "device") else next(unwrapped_model.parameters()).device
        vocab_size = unwrapped_model.config.vocab_size
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # Convert inputs to tensors
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Build canvas padded to max length
        x_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(inputs):
            x_ids[i, : seq_lens[i]] = t

        # Build attention mask
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        for i, L in enumerate(seq_lens):
            attention_mask[i, :L] = 1

        # Identify masked positions
        mask_positions = x_ids == mask_id

        # Get embedding layer
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.model.embed_tokens

        # Initialize sphere representation
        # Non-masked positions: one-hot -> sphere
        non_mask_onehot = F.one_hot(
            torch.where(mask_positions, torch.zeros_like(x_ids), x_ids),
            num_classes=vocab_size,
        ).float()
        x_sphere = simplex_to_sphere(non_mask_onehot)

        # Masked positions: uniform prior
        prior_sample = uniform_prior((B, T, vocab_size), device=device)
        x_sphere = torch.where(
            mask_positions.unsqueeze(-1).expand_as(x_sphere),
            prior_sample,
            x_sphere,
        )

        # Get discrete embeddings for non-masked positions (fixed throughout)
        context_embeds = embed_layer(x_ids)  # (B, T, D)

        # Run flow integration
        x_sphere, histories = self.flow_integrate(
            x_sphere=x_sphere,
            flow_mask=mask_positions,
            context_embeds=context_embeds,
            attention_mask=attention_mask,
            config=config,
            steps=steps,
            temperature=temperature,
            inference_scaling=inference_scaling,
            return_histories=return_dict,
        )

        # Convert final sphere points to tokens
        final_probs = sphere_to_simplex(x_sphere)
        final_tokens = final_probs.argmax(dim=-1)

        # Merge original tokens with filled tokens
        output_ids = x_ids.clone()
        output_ids[mask_positions] = final_tokens[mask_positions]

        if return_dict:
            return SamplerOutput(sequences=output_ids, histories=histories)
        return output_ids
