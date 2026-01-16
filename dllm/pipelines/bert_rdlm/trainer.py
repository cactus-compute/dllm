"""
BertRDLMTrainer: Trainer class for BERT with RDLM objective.

Extends BertSFMTrainer with RDLM-specific:
- Prior distributions (masked, mixture)
- Riemannian normal interpolation
- Geometric noise schedule
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import transformers

from dllm.utils.configs import TrainingArguments
from dllm.core.trainers.utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback
from dllm.pipelines.bert_sfm.geodesic_utils import (
    exp_map, exp_map_inplace, log_map, make_tangent, uniform_prior,
    simplex_to_sphere, sphere_to_simplex,
    TimeEmbedding,
)
from dllm.pipelines.bert_sfm.trainer import geodesic_interpolant_to_onehot

from .rdlm_utils import (
    RDLMSchedule,
    RDLMScheduleConfig,
    RDLMPriorType,
    get_rdlm_prior,
    rdlm_interpolant,
    expected_logmap_to_onehots,
    compute_target_drift,
)


@dataclass
class BertRDLMTrainerConfig(TrainingArguments):
    """Configuration for BertRDLMTrainer."""
    # Prior configuration
    prior_type: str = "mixture"  # "uniform", "masked", or "mixture"
    mixing_prob: float = 0.5    # Probability of uniform state in mixture prior
    mask_idx: int = -1          # Index of mask token (-1 = use tokenizer's mask_token_id)
    add_mask_token: bool = True  # True = add extra mask dimension (RDLM default)
    init_lambda: Optional[float] = None
    mix_type: str = "step"
    mix_step_thr: float = 0.0

    # Schedule configuration (defaults match RDLM paper)
    schedule_type: str = "geometric"
    sigma_0: float = 0.001  # beta_0 in RDLM
    sigma_T: float = 0.2    # beta_f in RDLM (paper uses 0.2, not 1.0)
    n_time_steps: int = 10000  # preprocess_steps in RDLM paper
    preprocess_dims: int = 2 ** 14
    rho_scale: float = 1.0
    weight_type: str = "step"
    weight_left: float = 0.3
    weight_right: float = 0.75
    weight_lb: float = 1e-4
    weight_ub: float = 1.0

    # Interpolation
    use_riemannian_normal: bool = True

    # Loss
    loss_type: str = "ce"  # "ce" or "mse"
    loss_norm_type: str = "token"  # "batch", "sequence", "token"

    # Embedding
    embed_type: str = "spherical"  # "spherical" or "simplex"

    # Time sampling
    time_eps: float = 1e-4  # Avoid t=0 and t=1

    # Evaluation - RDLM uses 256 for text8, 1000 for lm1b
    eval_integration_steps: int = 256

    # Use simplified eval that matches training (single forward pass at random t)
    # instead of full integration. Useful for debugging.
    eval_simple: bool = False

    # Use stochastic (Euler-Maruyama) integration for eval
    # NOTE: For large vocabularies (BERT ~30k), stochastic=False is recommended
    # because the noise-to-signal ratio scales with sqrt(vocab_size).
    eval_stochastic: bool = False  # Changed default: use deterministic for large vocab

    # Noise scaling for large vocabularies (only used when eval_stochastic=True)
    # Scales noise by sqrt(reference_vocab_size / actual_vocab_size)
    eval_noise_scaling: bool = True
    eval_reference_vocab_size: int = 27  # text8 vocab size

    # Time embedding
    use_time_embedding: bool = False
    time_embedding_scale: float = 30.0

    # Self-consistency: expose model to off-bridge states during training
    self_consistency_prob: float = 0.1
    self_consistency_noise_scale: float = 0.1


class BertRDLMTrainer(transformers.Trainer):
    """
    BERT trainer with RDLM (Riemannian Diffusion Language Model) objective.

    Key differences from BertSFMTrainer:
    1. Supports masked and mixture priors (not just uniform)
    2. Uses Riemannian normal approximation for simulation-free training
    3. Precomputes α_t and ρ_t parameters
    4. Default geometric noise schedule
    """

    def __init__(
        self,
        model,
        args: BertRDLMTrainerConfig,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)

        # Store RDLM config from args
        self.prior_type = args.prior_type
        self.mixing_prob = args.mixing_prob
        self.add_mask_token = args.add_mask_token
        self.init_lambda = args.init_lambda
        self.mix_type = args.mix_type
        self.mix_step_thr = args.mix_step_thr
        self.preprocess_dims = args.preprocess_dims
        self.rho_scale = args.rho_scale
        self.weight_type = args.weight_type
        self.weight_left = args.weight_left
        self.weight_right = args.weight_right
        self.weight_lb = args.weight_lb
        self.weight_ub = args.weight_ub
        self.use_riemannian_normal = args.use_riemannian_normal
        self.loss_type = args.loss_type
        self.loss_norm_type = args.loss_norm_type
        self.embed_type = args.embed_type
        self.time_eps = args.time_eps
        self.eval_integration_steps = args.eval_integration_steps
        self.eval_simple = args.eval_simple
        self.eval_stochastic = args.eval_stochastic
        self.eval_noise_scaling = args.eval_noise_scaling
        self.eval_reference_vocab_size = args.eval_reference_vocab_size
        self.use_time_embedding = args.use_time_embedding
        self.time_embedding_scale = args.time_embedding_scale
        self.self_consistency_prob = args.self_consistency_prob
        self.self_consistency_noise_scale = args.self_consistency_noise_scale

        # Get vocab size from model
        unwrapped_model = model.module if hasattr(model, 'module') else model
        if hasattr(unwrapped_model.config, 'vocab_size'):
            self.model_vocab_size = unwrapped_model.config.vocab_size
        else:
            self.model_vocab_size = kwargs.get('vocab_size', 30522)
        self.vocab_size = self.model_vocab_size
        self.rdlm_vocab_size = self.model_vocab_size + (1 if self.add_mask_token else 0)

        # Set mask index
        if self.add_mask_token:
            self.mask_idx = self.rdlm_vocab_size - 1
        else:
            if args.mask_idx != -1:
                self.mask_idx = args.mask_idx
            else:
                tokenizer_mask = getattr(self.processing_class, "mask_token_id", None)
                self.mask_idx = tokenizer_mask if tokenizer_mask is not None else self.model_vocab_size - 1

        # Initialize RDLM schedule with precomputed values
        # Note: We defer device placement until first compute_loss call
        self.rdlm_schedule = None
        self.schedule_config = RDLMScheduleConfig(
            schedule_type=args.schedule_type,
            sigma_0=args.sigma_0,
            sigma_T=args.sigma_T,
            n_time_steps=args.n_time_steps,
            prior_type=args.prior_type,
            add_mask_token=self.add_mask_token,
            init_lambda=self.init_lambda,
            mix_type=self.mix_type,
            mix_step_thr=self.mix_step_thr,
            rho_scale=self.rho_scale,
            preprocess_dims=self.preprocess_dims,
            weight_type=self.weight_type,
            weight_left=self.weight_left,
            weight_right=self.weight_right,
            weight_lb=self.weight_lb,
            weight_ub=self.weight_ub,
        )

        # Time embedding (initialized lazily)
        self.time_embedding = None

        # Metrics
        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)

    def _ensure_schedule(self, device: torch.device):
        """Ensure RDLM schedule is initialized and on correct device."""
        if self.rdlm_schedule is None:
            self.rdlm_schedule = RDLMSchedule(
                config=self.schedule_config,
                device=device,
                precompute=True,
                manifold_dim=self.rdlm_vocab_size - 1,
            )
        elif self.rdlm_schedule.device != device:
            self.rdlm_schedule = self.rdlm_schedule.to(device)

    def get_prior_samples(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from RDLM prior distribution.

        Args:
            shape: (batch_size, seq_len, vocab_size)
            device: torch device
            dtype: tensor dtype
            t: optional time values for time-dependent mixture

        Returns:
            x0: [B, L, D] prior samples on hypersphere
        """
        return get_rdlm_prior(
            prior_type=self.prior_type,
            shape=shape,
            device=device,
            dtype=dtype,
            t=t,
            mask_idx=self.mask_idx,
            mixing_prob=self.mixing_prob,
            mix_type=self.mix_type,
            mix_step_thr=self.mix_step_thr,
            init_lambda=self.init_lambda,
            add_mask_token=self.add_mask_token,
        )

    def interpolate(
        self,
        x0: torch.Tensor,
        target_indices: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interpolated samples using RDLM method.

        Args:
            x0: [B, L, D] prior samples
            target_indices: [B, L] target token indices
            t: [B] time values

        Returns:
            xt: [B, L, D] interpolated samples on sphere
        """
        if self.use_riemannian_normal:
            return rdlm_interpolant(
                x0=x0,
                target_indices=target_indices,
                t=t,
                schedule=self.rdlm_schedule,
                mask_idx=self.mask_idx,
                add_noise=True,
            )
        else:
            # Fall back to geodesic interpolation
            return geodesic_interpolant_to_onehot(x0, target_indices, t)

    def _add_tangent_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        loss_mask: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        """
        Add tangent-space noise at xt to expose the model to off-bridge states.

        Memory-optimized version using in-place operations.

        Args:
            xt: Current state on sphere, shape (B, L, V)
            t: Time values, shape (B,)
            loss_mask: Boolean mask for flow positions (True = flow)
            noise_scale: Base noise scale

        Returns:
            Perturbed state on sphere, shape (B, L, V)
        """
        b, l, _ = xt.shape

        # Generate noise and project to tangent space in-place
        noise = torch.randn_like(xt)
        # make_tangent: v - <p, v> * p
        dot_pv = (xt * noise).sum(dim=-1, keepdim=True)
        noise.sub_(dot_pv * xt)  # noise is now tangent vector
        del dot_pv

        # Scale noise in-place: tangent * t * noise_scale
        t_expanded = t.view(b, 1, 1)
        noise.mul_(t_expanded * noise_scale)

        # Use in-place exp_map
        xt_noisy = exp_map_inplace(xt, noise)  # noise tensor now contains result
        del noise

        # Normalize in-place
        norm = xt_noisy.norm(dim=-1, keepdim=True).clamp_(min=1e-8)
        xt_noisy.div_(norm)
        del norm

        # Apply mask: only update flow positions
        flow_mask_expanded = loss_mask.unsqueeze(-1)
        return torch.where(flow_mask_expanded, xt_noisy, xt)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ):
        """
        Compute RDLM training loss (memory-optimized).

        Args:
            model: the model being trained
            inputs: dict with "input_ids" [B, L]
            return_outputs: whether to return model outputs

        Returns:
            loss or (loss, outputs) if return_outputs=True
        """
        assert self.processing_class.padding_side == "right"

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        B, L = input_ids.shape
        device = input_ids.device

        # Get the actual model device (may differ from input device due to HF Trainer)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        model_device = next(unwrapped_model.parameters()).device
        if device != model_device:
            device = model_device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        # Ensure schedule is initialized and on correct device
        self._ensure_schedule(device)

        # Positions where we compute loss (not -100)
        loss_mask = labels != -100  # [b, l]

        # Get embedding layer and compute dtype
        if hasattr(unwrapped_model, "get_input_embeddings"):
            embed_layer = unwrapped_model.get_input_embeddings()
        else:
            embed_layer = unwrapped_model.model.embed_tokens
        compute_dtype = embed_layer.weight.dtype

        # Sample time (importance-weighted for RDLM, uniform otherwise)
        eps = self.time_eps
        if model.training and self.rdlm_schedule.weight_type != "default":
            t = self.rdlm_schedule.importance_weighted_time((B,), device)
            t = t.clamp(max=1 - eps)
        else:
            t = (1 - eps) * torch.rand(B, device=device)

        # Sample from prior and interpolate in one block to minimize peak memory
        x0 = self.get_prior_samples((B, L, self.rdlm_vocab_size), device, compute_dtype, t)
        xt = self.interpolate(x0, input_ids, t)
        del x0  # Free memory immediately

        # Keep prompt positions clean (not noised) - in-place where possible
        if not loss_mask.all():
            prompt_mask = ~loss_mask
            xt[prompt_mask] = 0
            xt[prompt_mask] = xt[prompt_mask].scatter(-1, input_ids[prompt_mask].unsqueeze(-1), 1.0)

        # Self-consistency: optionally perturb xt to expose off-bridge states
        if (
            self.self_consistency_prob > 0
            and model.training
            and random.random() < self.self_consistency_prob
        ):
            xt = self._add_tangent_noise(xt, t, loss_mask, self.self_consistency_noise_scale)

        # Compute soft embeddings (memory-optimized)
        x_embed = xt if self.embed_type == "spherical" else sphere_to_simplex(xt)
        if self.add_mask_token and x_embed.shape[-1] > self.model_vocab_size:
            x_embed_model = x_embed[..., :self.model_vocab_size]
        else:
            x_embed_model = x_embed
        soft_embeddings = torch.matmul(x_embed_model.to(compute_dtype), embed_layer.weight)
        del x_embed_model  # Free the slice/reference

        # Free xt and x_embed if we don't need them for MSE loss
        if self.loss_type == "ce":
            del xt, x_embed

        # Add time embedding if enabled
        if self.use_time_embedding:
            if self.time_embedding is None:
                hidden_size = embed_layer.weight.shape[1]
                self.time_embedding = TimeEmbedding(
                    hidden_size=hidden_size,
                    scale=self.time_embedding_scale,
                ).to(device=device, dtype=compute_dtype)
            time_emb = self.time_embedding(t)
            soft_embeddings = soft_embeddings + time_emb.unsqueeze(1).to(soft_embeddings.dtype)
            del time_emb

        # Forward pass
        outputs = model(
            inputs_embeds=soft_embeddings,
            attention_mask=attention_mask,
        )
        del soft_embeddings  # Free memory before logits allocation
        logits = outputs.logits  # [B, L, V]
        if not return_outputs:
            del outputs

        # Compute loss based on loss type
        if self.loss_type == "ce":
            # Cross-entropy loss
            token_loss = F.cross_entropy(
                logits.transpose(1, 2),  # [B, V, L]
                input_ids,  # [B, L]
                reduction='none'  # [B, L]
            )
            del logits

        elif self.loss_type == "mse":
            # MSE drift matching loss (ELBO)
            drift_coeff = self.rdlm_schedule.get_gamma(t)
            target_drift = compute_target_drift(xt, input_ids, drift_coeff, self.rdlm_vocab_size)

            probs = F.softmax(logits.to(torch.float32), dim=-1)
            del logits
            if probs.shape[-1] < xt.shape[-1]:
                pad = xt.shape[-1] - probs.shape[-1]
                probs = torch.cat([probs, probs.new_zeros(*probs.shape[:-1], pad)], dim=-1)

            predicted_drift = expected_logmap_to_onehots(xt, probs, positive_orthant=False)
            del probs
            predicted_drift = predicted_drift * drift_coeff.view(-1, 1, 1)
            predicted_drift = make_tangent(xt, predicted_drift)

            diff = predicted_drift - target_drift
            del predicted_drift, target_drift
            token_loss = 0.5 * diff.square().sum(dim=-1)
            del diff

            beta_t = self.rdlm_schedule.get_sigma(t)
            token_loss = token_loss / beta_t.view(-1, 1).clamp(min=1e-8)
            del xt

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Update metrics with raw loss (before importance weighting and masking)
        self.meter.update(
            split="train" if model.training else "eval",
            value=(token_loss * loss_mask.float()).detach(),
            weight=loss_mask.float().detach(),
        )

        # Importance weight for RDLM (applied after metrics update to keep metrics unweighted)
        weight = self.rdlm_schedule.importance_weight(t, model.training).view(B, 1)
        token_loss = token_loss * weight

        # Apply loss mask
        token_loss = token_loss * loss_mask.float()

        # Normalize loss
        if self.loss_norm_type == "token":
            loss = token_loss.sum() / loss_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            loss = token_loss.sum() / (loss_mask.sum(-1, keepdim=True).clamp_min(1) * B)
            loss = loss.sum()
        elif self.loss_norm_type == "batch":
            loss = token_loss.sum() / B
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")

        if return_outputs:
            return loss, outputs
        return loss

    @torch.no_grad()
    def prediction_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """
        Evaluation step for RDLM.

        If eval_simple=True: single forward pass at random t (matches training)
        If eval_simple=False: full flow integration (measures generation quality)
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        B, L = input_ids.shape
        unwrapped_model = model.module if hasattr(model, 'module') else model
        model_vocab_size = unwrapped_model.config.vocab_size
        rdlm_vocab_size = self.rdlm_vocab_size
        device = input_ids.device

        # Get the actual model device (may differ from input device due to HF Trainer)
        model_device = next(unwrapped_model.parameters()).device
        if device != model_device:
            device = model_device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        # Ensure schedule is initialized
        self._ensure_schedule(device)

        # Positions where we compute loss
        loss_mask = labels != -100

        # Get embedding layer
        if hasattr(unwrapped_model, "get_input_embeddings"):
            embed_layer = unwrapped_model.get_input_embeddings()
        else:
            embed_layer = unwrapped_model.model.embed_tokens
        compute_dtype = embed_layer.weight.dtype

        if self.eval_simple:
            # Simple eval: single forward pass at random t (matches training)
            eps = self.time_eps
            t = (1 - eps) * torch.rand(B, device=device)

            # Sample from prior
            x0 = self.get_prior_samples((B, L, rdlm_vocab_size), device, compute_dtype, t)

            # Get interpolated samples
            xt = self.interpolate(x0, input_ids, t)

            # Keep prompt positions clean
            if not loss_mask.all():
                prompt_mask = ~loss_mask
                xt[prompt_mask] = 0
                prompt_indices = input_ids[prompt_mask].unsqueeze(-1)
                xt[prompt_mask] = xt[prompt_mask].scatter(-1, prompt_indices, 1.0)

            # Compute soft embeddings
            x_embed = xt if self.embed_type == "spherical" else sphere_to_simplex(xt)
            if self.add_mask_token and x_embed.shape[-1] > self.model_vocab_size:
                x_embed_model = x_embed[..., :self.model_vocab_size]
            else:
                x_embed_model = x_embed
            soft_embeddings = torch.matmul(x_embed_model.to(compute_dtype), embed_layer.weight)

            # Add time embedding if enabled
            if self.use_time_embedding and self.time_embedding is not None:
                time_emb = self.time_embedding(t)
                soft_embeddings = soft_embeddings + time_emb.unsqueeze(1).to(soft_embeddings.dtype)

            # Forward pass
            outputs = model(
                inputs_embeds=soft_embeddings,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # CE loss
            token_loss = F.cross_entropy(
                logits.transpose(1, 2),
                input_ids,
                reduction='none'
            )
            token_loss = token_loss * loss_mask.float()

        else:
            # Full integration eval
            from dllm.pipelines.bert_rdlm.sampler import BertRDLMSampler, BertRDLMSamplerConfig

            # Initialize sphere state
            prompt_onehot = F.one_hot(input_ids, num_classes=rdlm_vocab_size).float()
            x_sphere = simplex_to_sphere(prompt_onehot)

            # For flow positions: sample from RDLM prior
            prior_sample = self.get_prior_samples(
                (B, L, rdlm_vocab_size), device, compute_dtype, t=torch.zeros(B, device=device)
            )
            x_sphere = torch.where(
                loss_mask.unsqueeze(-1).expand_as(x_sphere),
                prior_sample,
                x_sphere,
            )

            # Get discrete embeddings for context
            context_embeds = embed_layer(input_ids)

            # Create sampler and config
            sampler = BertRDLMSampler(
                model=model,
                tokenizer=self.processing_class,
                rdlm_schedule=self.rdlm_schedule,
            )
            config = BertRDLMSamplerConfig(
                n_steps=self.eval_integration_steps,
                integrator="euler",
                prior_type=self.prior_type,
                mixing_prob=self.mixing_prob,
                mask_idx=self.mask_idx,
                add_mask_token=self.add_mask_token,
                schedule_type=self.schedule_config.schedule_type,
                sigma_0=self.schedule_config.sigma_0,
                sigma_T=self.schedule_config.sigma_T,
                temperature=0.0,
                embed_type=self.embed_type,
                stochastic=self.eval_stochastic,
                noise_scaling=self.eval_noise_scaling,
                reference_vocab_size=self.eval_reference_vocab_size,
                mix_type=self.mix_type,
                mix_step_thr=self.mix_step_thr,
            )

            # Run flow integration
            x_sphere = sampler.flow_integrate(
                x_sphere=x_sphere,
                flow_mask=loss_mask,
                context_embeds=context_embeds,
                attention_mask=attention_mask,
                config=config,
                time_embedding=self.time_embedding if self.use_time_embedding else None,
            )

            # Compute final loss
            final_probs = sphere_to_simplex(x_sphere)
            if final_probs.shape[-1] > model_vocab_size:
                final_probs = final_probs[..., :model_vocab_size]
            final_log_probs = torch.log(final_probs.clamp(min=1e-10))

            token_loss = F.nll_loss(
                final_log_probs.transpose(1, 2),
                input_ids,
                reduction="none",
            )
            token_loss = token_loss * loss_mask.float()

        # Update metrics
        self.meter.update(
            split="eval",
            value=token_loss.detach(),
            weight=loss_mask.float().detach(),
        )

        loss = token_loss.sum() / loss_mask.sum().clamp_min(1)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), None, labels.detach())
