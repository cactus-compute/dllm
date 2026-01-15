"""
BertRDLMSampler: Sampler class for BERT with RDLM objective.

Implements sampling from RDLM using Euler-Maruyama integration.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map,
    exp_map_inplace,
    log_map,
    log_map_inplace,
    make_tangent,
    simplex_to_sphere,
    sphere_to_simplex,
    uniform_prior,
    TimeEmbedding,
    expected_logmap_to_onehots,
)
from .rdlm_utils import (
    RDLMSchedule,
    RDLMPriorType,
    get_rdlm_prior,
    geometric_schedule,
    linear_schedule,
    cosine_schedule,
    bridge_gamma,
    expected_drift_from_probs,
)


@dataclass
class BertRDLMSamplerConfig(SamplerConfig):
    """Configuration for RDLM sampler."""
    # Integration (RDLM uses 256 for text8, 1000 for lm1b)
    n_steps: int = 256
    integrator: str = "euler"  # "euler" or "euler_maruyama"

    # Prior
    prior_type: str = "mixture"  # "uniform", "masked", "mixture"
    mixing_prob: float = 0.5
    mask_idx: int = -1

    # Schedule (defaults match RDLM paper)
    schedule_type: str = "geometric"  # "geometric", "linear", "cosine"
    sigma_0: float = 0.001  # beta_0 in RDLM
    sigma_T: float = 0.2    # beta_f in RDLM (paper uses 0.2)

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.0

    # Prediction type
    prediction_type: str = "endpoint"  # "endpoint" or "drift"

    # Embedding
    embed_type: str = "spherical"  # "spherical" or "simplex"

    # Whether to add stochastic noise during sampling (RDLM uses grw/Euler-Maruyama)
    stochastic: bool = True


@dataclass
class BertRDLMSampler(BaseSampler):
    """
    RDLM Sampler for BERT-style models.

    Generates samples by integrating the reverse-time SDE from t=0 to t=1.
    """
    rdlm_schedule: Optional[RDLMSchedule] = None

    def __post_init__(self):
        """Initialize components after dataclass init."""
        pass

    def _get_sigma_fn(self, config: BertRDLMSamplerConfig):
        """Get sigma function based on config."""
        if config.schedule_type == "geometric":
            return lambda t: geometric_schedule(t, config.sigma_0, config.sigma_T)
        elif config.schedule_type == "linear":
            return lambda t: linear_schedule(t, config.sigma_0, config.sigma_T)
        elif config.schedule_type == "cosine":
            return lambda t: cosine_schedule(t, config.sigma_0, config.sigma_T)
        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")

    def _compute_tangent(
        self,
        x_sphere: torch.Tensor,
        flow_mask: torch.Tensor,
        context_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        config: BertRDLMSamplerConfig,
        t: torch.Tensor,
        temperature: float,
        embed_layer,
        time_embedding: Optional[TimeEmbedding] = None,
    ) -> torch.Tensor:
        """
        Compute tangent vector (velocity/drift) at current state.

        Args:
            x_sphere: current state on sphere [B, T, V]
            flow_mask: positions to update [B, T]
            context_embeds: embeddings for context [B, T, D]
            attention_mask: attention mask [B, T]
            config: sampler config
            t: current time (scalar or [B])
            temperature: sampling temperature
            embed_layer: model's embedding layer
            time_embedding: optional time embedding module

        Returns:
            tangent: tangent vector [B, T, V]
        """
        # Compute soft embeddings
        x_embed = x_sphere if config.embed_type == "spherical" else sphere_to_simplex(x_sphere)
        soft_embeddings = torch.matmul(
            x_embed.to(embed_layer.weight.dtype), embed_layer.weight
        )

        # Use discrete embeddings for context positions
        soft_embeddings[~flow_mask] = context_embeds[~flow_mask]

        # Add time embedding if provided
        if time_embedding is not None:
            batch_size = x_sphere.shape[0]
            t_batch = t.expand(batch_size) if t.dim() == 0 else t
            time_emb = time_embedding(t_batch)
            soft_embeddings = soft_embeddings + time_emb.unsqueeze(1).to(soft_embeddings.dtype)

        # Forward pass
        outputs = self.model(
            inputs_embeds=soft_embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Compute tangent based on prediction type
        if config.prediction_type == "endpoint":
            # Model predicts distribution over endpoints (tokens)
            # Compute expected log-map toward one-hots weighted by probabilities
            # This matches RDLM's weighted_sum operation
            probs = F.softmax(logits, dim=-1)
            # Use positive_orthant=False to handle full sphere (RDLM style)
            tangent = expected_logmap_to_onehots(x_sphere, probs, positive_orthant=False)
        elif config.prediction_type == "drift":
            tangent = make_tangent(x_sphere, logits)
        else:
            raise ValueError(f"Unknown prediction_type: {config.prediction_type}")

        return tangent

    def flow_integrate(
        self,
        x_sphere: torch.Tensor,
        flow_mask: torch.Tensor,
        context_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        config: BertRDLMSamplerConfig,
        time_embedding: Optional[TimeEmbedding] = None,
        return_histories: bool = False,
    ) -> torch.Tensor:
        """
        Integrate flow from t=0 to t=1.

        Args:
            x_sphere: initial state on sphere [B, T, V]
            flow_mask: positions to update [B, T]
            context_embeds: embeddings for context [B, T, D]
            attention_mask: attention mask [B, T]
            config: sampler config
            time_embedding: optional time embedding
            return_histories: whether to return intermediate states

        Returns:
            x_sphere: final state on sphere [B, T, V]
        """
        # Get embedding layer
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.model.embed_tokens

        device = x_sphere.device
        n_steps = config.n_steps
        dt = 1.0 / n_steps

        sigma_fn = self._get_sigma_fn(config)

        histories = [] if return_histories else None
        if return_histories:
            histories.append(sphere_to_simplex(x_sphere).argmax(dim=-1).clone())

        # Integration loop
        for step in range(n_steps):
            t = step * dt
            t_tensor = torch.tensor(t, device=device)

            # Compute tangent direction
            tangent = self._compute_tangent(
                x_sphere, flow_mask, context_embeds, attention_mask,
                config, t_tensor, config.temperature, embed_layer, time_embedding
            )

            # Scale by drift coefficient and dt
            # RDLM drift_coeff depends on schedule type:
            # - Geometric with sigma_0 == sigma_T: drift_coeff = 1 / (1-t)
            # - Geometric with sigma_0 != sigma_T: drift_coeff = log(r) / (r^(1-t) - 1)
            if config.prediction_type == "endpoint":
                if abs(config.sigma_0 - config.sigma_T) < 1e-8:
                    # Constant sigma case
                    drift_coeff = 1.0 / max(1 - t, 1e-4)
                else:
                    # General geometric case
                    r = config.sigma_T / config.sigma_0
                    drift_coeff = math.log(r) / max(r ** (1 - t) - 1, 1e-8)
                tangent = tangent * drift_coeff
                # CRITICAL: RDLM applies to_tangent AFTER scaling by drift coefficient
                # This ensures the drift stays on the tangent plane after scaling
                tangent = make_tangent(x_sphere, tangent)
                tangent = tangent * dt
            else:
                tangent = tangent * dt

            # Add stochastic noise if enabled (Euler-Maruyama)
            # RDLM uses: sqrt(beta(t)) * z * sqrt(dt) where beta = sigma in our notation
            # diffusion = sqrt(beta(t)), so total noise scale is diffusion * sqrt(dt) = sqrt(beta(t) * dt)
            if config.stochastic:
                sigma_t = sigma_fn(t_tensor)
                diffusion = sigma_t.sqrt()  # RDLM's diffusion coefficient
                noise = torch.randn_like(x_sphere)
                # Don't project noise yet - RDLM doesn't project noise before adding
                tangent = tangent + diffusion * noise * math.sqrt(dt)

            # CRITICAL: Project final tangent vector before exp_map
            # RDLM's exp() internally calls to_tangent before computing the exponential
            tangent = make_tangent(x_sphere, tangent)

            # Take step via exponential map
            x_new = exp_map(x_sphere, tangent)

            # Project to sphere for numerical stability
            x_new = x_new / torch.norm(x_new, dim=-1, keepdim=True).clamp(min=1e-8)

            # Only update flow positions
            x_sphere = torch.where(
                flow_mask.unsqueeze(-1).expand_as(x_sphere),
                x_new,
                x_sphere,
            )

            if return_histories:
                histories.append(sphere_to_simplex(x_sphere).argmax(dim=-1).clone())

        if return_histories:
            return x_sphere, histories
        return x_sphere

    @torch.no_grad()
    def sample(
        self,
        inputs: List[torch.Tensor],
        config: Optional[BertRDLMSamplerConfig] = None,
        time_embedding: Optional[TimeEmbedding] = None,
        **kwargs,
    ) -> SamplerOutput:
        """
        Generate text using RDLM sampling.

        Args:
            inputs: list of input prompts (token tensors)
            config: sampler configuration
            time_embedding: optional time embedding module
            **kwargs: override specific config parameters

        Returns:
            SamplerOutput with generated sequences
        """
        if config is None:
            config = BertRDLMSamplerConfig()

        # Override config with kwargs
        n_steps = kwargs.get("n_steps", config.n_steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        temperature = kwargs.get("temperature", config.temperature)
        return_dict = kwargs.get("return_dict", getattr(config, "return_dict", True))

        # Setup
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        device = next(unwrapped_model.parameters()).device
        vocab_size = unwrapped_model.config.vocab_size
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # Convert inputs to tensors
        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs]

        B = len(inputs)
        prompt_lens = [p.shape[0] for p in inputs]
        T = max_new_tokens + max(prompt_lens)

        # Initialize canvas
        x_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        for i, p in enumerate(inputs):
            x_ids[i, :prompt_lens[i]] = p

        # Attention mask
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        for i, pl in enumerate(prompt_lens):
            attention_mask[i, :pl + max_new_tokens] = 1

        # Generation mask
        gen_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for i, pl in enumerate(prompt_lens):
            gen_mask[i, pl:pl + max_new_tokens] = True

        # Get embedding layer
        if hasattr(self.model, "get_input_embeddings"):
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.model.embed_tokens
        compute_dtype = embed_layer.weight.dtype

        # Determine mask index
        mask_idx = config.mask_idx if config.mask_idx != -1 else vocab_size - 1

        # Initialize sphere state
        # Prompt positions: one-hot on sphere
        prompt_onehot = F.one_hot(x_ids, num_classes=vocab_size).float()
        x_sphere = simplex_to_sphere(prompt_onehot)

        # Generation positions: sample from prior
        gen_shape = (B, max_new_tokens, vocab_size)
        gen_prior = get_rdlm_prior(
            prior_type=config.prior_type,
            shape=gen_shape,
            device=device,
            dtype=compute_dtype,
            mask_idx=mask_idx,
            mixing_prob=config.mixing_prob,
        )
        for i, pl in enumerate(prompt_lens):
            x_sphere[i, pl:pl + max_new_tokens] = gen_prior[i]

        # Get context embeddings
        context_embeds = embed_layer(x_ids)

        # Create config with updated parameters
        run_config = BertRDLMSamplerConfig(
            n_steps=n_steps,
            integrator=config.integrator,
            prior_type=config.prior_type,
            mixing_prob=config.mixing_prob,
            mask_idx=mask_idx,
            schedule_type=config.schedule_type,
            sigma_0=config.sigma_0,
            sigma_T=config.sigma_T,
            temperature=temperature,
            prediction_type=config.prediction_type,
            embed_type=config.embed_type,
            stochastic=config.stochastic,
        )

        # Run flow integration
        result = self.flow_integrate(
            x_sphere=x_sphere,
            flow_mask=gen_mask,
            context_embeds=context_embeds,
            attention_mask=attention_mask,
            config=run_config,
            time_embedding=time_embedding,
            return_histories=return_dict,
        )

        if return_dict:
            x_sphere, histories = result
        else:
            x_sphere = result
            histories = None

        # Convert to tokens
        final_probs = sphere_to_simplex(x_sphere)
        final_tokens = final_probs.argmax(dim=-1)

        # Merge prompt with generated tokens
        output_ids = x_ids.clone()
        output_ids[gen_mask] = final_tokens[gen_mask]

        if return_dict:
            return SamplerOutput(sequences=output_ids, histories=histories)
        return output_ids

    @torch.no_grad()
    def infill(
        self,
        inputs: List[torch.Tensor],
        config: Optional[BertRDLMSamplerConfig] = None,
        time_embedding: Optional[TimeEmbedding] = None,
        **kwargs,
    ) -> SamplerOutput:
        """
        Fill in masked positions within input sequences.

        Args:
            inputs: list of sequences with mask tokens
            config: sampler configuration
            time_embedding: optional time embedding module
            **kwargs: override specific config parameters

        Returns:
            SamplerOutput with filled sequences
        """
        if config is None:
            config = BertRDLMSamplerConfig()

        # Override config with kwargs
        n_steps = kwargs.get("n_steps", config.n_steps)
        temperature = kwargs.get("temperature", config.temperature)
        return_dict = kwargs.get("return_dict", getattr(config, "return_dict", True))

        # Setup
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        device = next(unwrapped_model.parameters()).device
        vocab_size = unwrapped_model.config.vocab_size
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # Convert inputs
        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Build canvas
        x_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(inputs):
            x_ids[i, :seq_lens[i]] = t

        # Attention mask
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
        compute_dtype = embed_layer.weight.dtype

        # Determine mask index
        mask_idx = config.mask_idx if config.mask_idx != -1 else vocab_size - 1

        # Initialize sphere state
        # Non-masked positions: one-hot on sphere
        non_mask_ids = torch.where(mask_positions, torch.zeros_like(x_ids), x_ids)
        non_mask_onehot = F.one_hot(non_mask_ids, num_classes=vocab_size).float()
        x_sphere = simplex_to_sphere(non_mask_onehot)

        # Masked positions: sample from prior
        prior_sample = get_rdlm_prior(
            prior_type=config.prior_type,
            shape=(B, T, vocab_size),
            device=device,
            dtype=compute_dtype,
            mask_idx=mask_idx,
            mixing_prob=config.mixing_prob,
        )
        x_sphere = torch.where(
            mask_positions.unsqueeze(-1).expand_as(x_sphere),
            prior_sample,
            x_sphere,
        )

        # Context embeddings
        context_embeds = embed_layer(x_ids)

        # Create run config
        run_config = BertRDLMSamplerConfig(
            n_steps=n_steps,
            integrator=config.integrator,
            prior_type=config.prior_type,
            mixing_prob=config.mixing_prob,
            mask_idx=mask_idx,
            schedule_type=config.schedule_type,
            sigma_0=config.sigma_0,
            sigma_T=config.sigma_T,
            temperature=temperature,
            prediction_type=config.prediction_type,
            embed_type=config.embed_type,
            stochastic=config.stochastic,
        )

        # Run flow integration
        result = self.flow_integrate(
            x_sphere=x_sphere,
            flow_mask=mask_positions,
            context_embeds=context_embeds,
            attention_mask=attention_mask,
            config=run_config,
            time_embedding=time_embedding,
            return_histories=return_dict,
        )

        if return_dict:
            x_sphere, histories = result
        else:
            x_sphere = result
            histories = None

        # Convert to tokens
        final_probs = sphere_to_simplex(x_sphere)
        final_tokens = final_probs.argmax(dim=-1)

        # Merge
        output_ids = x_ids.clone()
        output_ids[mask_positions] = final_tokens[mask_positions]

        if return_dict:
            return SamplerOutput(sequences=output_ids, histories=histories)
        return output_ids
