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
from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map,
    exp_map_inplace,
    expected_logmap_to_onehots,
    log_map,
    log_map_inplace,
    make_tangent,
    project_to_sphere,
    simplex_to_sphere,
    sphere_to_simplex,
    uniform_prior,
    linear_schedule,
    cosine_schedule,
    TimeEmbedding,
)


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
    embed_type: str = "spherical"  # "spherical" or "simplex"
    prediction_type: str = "endpoint"  # "endpoint" (CE) or "velocity" (MSE)
    # Step weight capping to prevent blow-up near t=1
    # Set to 0 to disable capping, otherwise caps step_weight at this multiple of dt
    step_weight_cap: float = 0.0  # 0 = no cap, e.g. 4.0 = cap at 4x dt
    # Use expected_logmap_to_onehots instead of log_map(x_t, sqrt(probs))
    # This is mathematically correct for computing the expected direction
    # toward a categorical distribution, accounting for log_map nonlinearity.
    # Significantly more stable and accurate than the legacy sqrt(probs) approach.
    # Default: True (recommended). Set to False only for legacy compatibility.
    use_expected_logmap: bool = True
    # Integrator type for flow integration
    # "euler" = standard Euler method (1 forward pass per step)
    # "rk2" = Midpoint method / RK2 (2 forward passes per step, more accurate)
    integrator_type: str = "euler"


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

    def _compute_tangent(
        self,
        x_sphere: torch.Tensor,
        flow_mask: torch.Tensor,
        context_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        config: BertSFMSamplerConfig,
        t: torch.Tensor,
        temperature: float,
        embed_layer,
        time_embedding: TimeEmbedding | None = None,
    ) -> torch.Tensor:
        """
        Compute the tangent vector (velocity) at the current sphere state.

        This is the core computation shared by all integrators: given x_sphere at time t,
        run the model and compute the tangent direction toward the predicted endpoint.

        Args:
            x_sphere: Current sphere state, shape (B, T, V).
            flow_mask: Boolean mask for flow positions (B, T).
            context_embeds: Discrete embeddings for context positions (B, T, D).
            attention_mask: Attention mask for the model (B, T).
            config: Sampler configuration.
            t: Current time, scalar tensor.
            temperature: Temperature for logits.
            embed_layer: Model's embedding layer.
            time_embedding: Optional time embedding module.

        Returns:
            Tangent vector (unnormalized velocity), shape (B, T, V).
        """
        # Compute soft embeddings from current sphere state
        x_embed = x_sphere if config.embed_type == "spherical" else sphere_to_simplex(x_sphere)
        soft_embeddings = torch.matmul(
            x_embed.to(embed_layer.weight.dtype), embed_layer.weight
        )
        if x_embed is not x_sphere:
            del x_embed

        # Use discrete embeddings for context positions
        soft_embeddings[~flow_mask] = context_embeds[~flow_mask]

        # Add time embedding if provided
        if time_embedding is not None:
            batch_size = x_sphere.shape[0]
            t_batch = t.expand(batch_size)
            time_emb = time_embedding(t_batch)
            soft_embeddings = soft_embeddings + time_emb.unsqueeze(1).to(soft_embeddings.dtype)

        # Model forward pass
        outputs = self.model(
            inputs_embeds=soft_embeddings,
            attention_mask=attention_mask,
        )
        del soft_embeddings
        logits = outputs.logits
        del outputs

        # Apply temperature
        if temperature > 0:
            logits.div_(temperature)

        # Compute tangent based on prediction type
        prediction_type = getattr(config, "prediction_type", "endpoint")

        if prediction_type == "endpoint":
            probs = F.softmax(logits, dim=-1)
            del logits

            use_expected_logmap = getattr(config, "use_expected_logmap", True)
            if use_expected_logmap:
                tangent = expected_logmap_to_onehots(x_sphere, probs)
                del probs
            else:
                x_1_pred = probs.sqrt_()
                del probs
                tangent = log_map_inplace(x_sphere, x_1_pred)

        elif prediction_type == "velocity":
            tangent = make_tangent(x_sphere, logits)
            del logits
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        return tangent

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
        time_embedding: TimeEmbedding | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Core flow integration loop on the Fisher-Rao manifold.

        Integrates from t=0 to t=1, updating only positions where flow_mask=True.

        Supports multiple integrators:
        - "euler": Standard Euler method (1 forward pass per step)
        - "rk2": Midpoint method (2 forward passes per step, more accurate)

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
            time_embedding: Optional TimeEmbedding module for conditioning on timestep.

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

        integrator_type = getattr(config, "integrator_type", "euler")
        prediction_type = getattr(config, "prediction_type", "endpoint")
        step_weight_cap = getattr(config, "step_weight_cap", 0.0)

        # Integration loop
        for step_idx in range(steps):
            t_curr = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            dt = t_next - t_curr

            if integrator_type == "euler":
                # Standard Euler: x_{n+1} = exp(x_n, tangent * step_weight)
                alpha_t, alpha_t_prime = self._get_schedule(t_curr.unsqueeze(0), config)

                tangent = self._compute_tangent(
                    x_sphere, flow_mask, context_embeds, attention_mask,
                    config, t_curr, temperature, embed_layer, time_embedding
                )

                if prediction_type == "endpoint":
                    step_weight = (alpha_t_prime * dt / (1 - alpha_t + 1e-5)) * inference_scaling
                    if step_weight_cap > 0:
                        step_weight = min(step_weight, step_weight_cap * dt)
                    tangent.mul_(step_weight)
                else:
                    tangent.mul_(dt * inference_scaling)

                x_sphere_new = exp_map_inplace(x_sphere, tangent)
                del tangent

            elif integrator_type == "rk2":
                # RK2 / Midpoint method:
                # 1. Compute k1 = tangent at (x_n, t_n)
                # 2. Take half step: x_mid = exp(x_n, k1 * step_weight/2)
                # 3. Compute k2 = tangent at (x_mid, t_n + dt/2)
                # 4. Take full step from x_n using k2: x_{n+1} = exp(x_n, k2 * step_weight)

                # Step 1: Compute k1 at current point
                alpha_t, alpha_t_prime = self._get_schedule(t_curr.unsqueeze(0), config)

                k1 = self._compute_tangent(
                    x_sphere, flow_mask, context_embeds, attention_mask,
                    config, t_curr, temperature, embed_layer, time_embedding
                )

                if prediction_type == "endpoint":
                    step_weight = (alpha_t_prime * dt / (1 - alpha_t + 1e-5)) * inference_scaling
                    if step_weight_cap > 0:
                        step_weight = min(step_weight, step_weight_cap * dt)
                else:
                    step_weight = dt * inference_scaling

                # Step 2: Take half step to get midpoint
                k1_half = k1.mul(step_weight * 0.5)
                del k1
                x_mid = exp_map(x_sphere, k1_half)
                del k1_half
                x_mid.div_(torch.norm(x_mid, dim=-1, keepdim=True).clamp(min=1e-8))

                # Step 3: Compute k2 at midpoint (at time t + dt/2)
                t_mid = t_curr + dt * 0.5
                alpha_t_mid, alpha_t_prime_mid = self._get_schedule(t_mid.unsqueeze(0), config)

                k2 = self._compute_tangent(
                    x_mid, flow_mask, context_embeds, attention_mask,
                    config, t_mid, temperature, embed_layer, time_embedding
                )
                del x_mid

                # Recompute step weight at midpoint for endpoint prediction
                if prediction_type == "endpoint":
                    step_weight_mid = (alpha_t_prime_mid * dt / (1 - alpha_t_mid + 1e-5)) * inference_scaling
                    if step_weight_cap > 0:
                        step_weight_mid = min(step_weight_mid, step_weight_cap * dt)
                    k2.mul_(step_weight_mid)
                else:
                    k2.mul_(dt * inference_scaling)

                # Step 4: Take full step from x_n using k2
                x_sphere_new = exp_map_inplace(x_sphere, k2)
                del k2

            else:
                raise ValueError(f"Unknown integrator_type: {integrator_type}")

            # Project to sphere
            x_sphere_new.div_(torch.norm(x_sphere_new, dim=-1, keepdim=True).clamp(min=1e-8))

            # Only update flow positions; keep context positions fixed
            x_sphere[flow_mask] = x_sphere_new[flow_mask]
            del x_sphere_new

            if return_histories:
                current_tokens = sphere_to_simplex(x_sphere).argmax(dim=-1)
                histories.append(current_tokens.clone())

        return x_sphere, histories if return_histories else None

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: BertSFMSamplerConfig | None = None,
        time_embedding: TimeEmbedding | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Generate text using Fisher-Rao flow matching.

        Integrates the learned flow from t=0 to t=1, starting from the prior
        and moving toward the predicted data distribution at each step.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            time_embedding: Optional TimeEmbedding module for conditioning on timestep.
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
            time_embedding=time_embedding,
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
        time_embedding: TimeEmbedding | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Fill in masked positions within input sequences.

        Uses flow matching to generate tokens at positions marked with mask_token.

        Args:
            inputs: List of sequences containing mask tokens to fill.
            config: Sampler configuration, or None to use defaults.
            time_embedding: Optional TimeEmbedding module for conditioning on timestep.
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
            time_embedding=time_embedding,
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
