#!/usr/bin/env python3
"""
Diagnostic training script for BERT SFM.

Runs training with detailed logging during evaluation:
1. Loss bucketed by time t (single-step, training-style)
2. Loss at each integration step (multi-step, eval-style)
3. Geodesic distance between ground-truth x_t and model's x_sphere

Usage:
    python examples/bert_sfm/diagnostic_train.py
"""

import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import accelerate
import torch
import torch.nn.functional as F
import transformers

import dllm
from dllm.pipelines.bert_sfm import BertSFMTrainer
from dllm.pipelines.bert_sfm.sampler import BertSFMSampler, BertSFMSamplerConfig
from dllm.pipelines.bert_sfm.geodesic_utils import (
    sphere_to_simplex,
    simplex_to_sphere,
    uniform_prior,
    geodesic_interpolant,
    exp_map,
    make_tangent,
    expected_logmap_to_onehots,
    log_map,
)
from dllm.pipelines.bert_sfm.trainer import geodesic_interpolant_to_onehot

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    load_preprocessed_data: bool = False
    skip_post_process: bool = False
    mask_prompt_loss: bool = True


@dataclass
class TrainingArguments(BertSFMTrainer.BertSFMConfig):
    output_dir: str = "models/diagnostic/spherical_ce_linear_uniform"
    group_by_length: bool = True
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 16
    schedule_type: str = "linear"
    loss_weight_type: str = "uniform"
    embed_type: str = "spherical"
    loss_type: str = "ce"
    save_only_model: bool = False
    # More frequent eval for diagnostics
    eval_steps: float = 0.05  # Every 5%
    # Use simple dt scaling instead of alpha_t_prime * dt / (1 - alpha_t)
    use_simple_dt: bool = False
    # Self-consistency training: exposes model to off-geodesic states during training
    # This addresses distribution mismatch between training (on-geodesic) and inference (off-geodesic)
    self_consistency_prob: float = 0.0  # 0.0 = disabled, try 0.5 for testing
    self_consistency_max_steps: int = 5  # Maximum simulation steps (for "simulate" mode)
    self_consistency_schedule: str = "constant"  # "constant" or "linear_ramp"
    # Self-consistency mode: "noise" (recommended) or "simulate"
    # "noise" adds tangent space noise - cheap and effective
    # "simulate" runs integration with model - expensive but more realistic
    self_consistency_mode: str = "noise"
    # Noise scale for "noise" mode (scaled by t, so more noise at later timesteps)
    self_consistency_noise_scale: float = 0.1
    # Step weight capping during evaluation to prevent blow-up near t=1
    # The formula alpha'(t)*dt/(1-alpha(t)) explodes as t->1, causing instability
    # 0 = no cap (default), e.g. 4.0 = cap step weight at 4x dt
    eval_step_weight_cap: float = 0.0
    # Geodesic loss: auxiliary loss to align CE training with inference
    # 0 = disabled, try 0.1-1.0 for moderate regularization
    geodesic_loss_weight: float = 0.0
    # Hybrid MSE loss: add velocity MSE as auxiliary loss alongside CE
    # More principled than geodesic loss - combines CE's strength with MSE's geometry
    # 0 = disabled (pure CE), try 0.1-1.0 for hybrid training
    mse_loss_weight: float = 0.0
    # Temperature for evaluation inference (logits / temperature)
    # temperature < 1.0 = sharper predictions, temperature > 1.0 = softer
    # 0 = no temperature scaling (equivalent to temperature=1.0)
    # Try values like 0.1, 0.2, 0.5 to sharpen predictions during integration
    eval_temperature: float = 0.0
    # Use expected_logmap_to_onehots instead of log_map(x_t, sqrt(probs))
    # This computes the mathematically correct expected direction toward a categorical
    # distribution, accounting for the nonlinearity of log_map on the sphere.
    # May reduce error accumulation during integration when predictions are uncertain.
    use_expected_logmap: bool = False


class DiagnosticBertSFMTrainer(BertSFMTrainer):
    """Extended trainer with diagnostic logging during evaluation."""

    def _init_diagnostic_accumulators(self):
        """Initialize accumulators for averaging metrics across eval batches."""
        steps = self.eval_integration_steps
        self._diag_bucket_losses = [[] for _ in range(10)]  # 10 time buckets
        self._diag_step_losses = [[] for _ in range(steps)]
        self._diag_step_distances = [[] for _ in range(steps)]
        self._diag_step_tvs = [[] for _ in range(steps)]  # Total variation per step
        self._diag_step_entropies = [[] for _ in range(steps)]  # Entropy per step
        self._diag_final_losses = []
        self._diag_batch_count = 0

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to aggregate diagnostic metrics across batches and log once."""
        # Initialize accumulators before evaluation
        self._init_diagnostic_accumulators()

        # Run the standard evaluation loop
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        # After all batches, compute and log averages
        if self.accelerator.is_main_process and self._diag_batch_count > 0:
            steps = self.eval_integration_steps

            # Average bucket losses
            avg_bucket_losses = [
                sum(self._diag_bucket_losses[i]) / len(self._diag_bucket_losses[i])
                if self._diag_bucket_losses[i] else 0.0
                for i in range(10)
            ]

            # Average step losses, distances, TVs, and entropies
            avg_step_losses = [
                sum(self._diag_step_losses[i]) / len(self._diag_step_losses[i])
                if self._diag_step_losses[i] else 0.0
                for i in range(steps)
            ]
            avg_step_distances = [
                sum(self._diag_step_distances[i]) / len(self._diag_step_distances[i])
                if self._diag_step_distances[i] else 0.0
                for i in range(steps)
            ]
            avg_step_tvs = [
                sum(self._diag_step_tvs[i]) / len(self._diag_step_tvs[i])
                if self._diag_step_tvs[i] else 0.0
                for i in range(steps)
            ]
            avg_step_entropies = [
                sum(self._diag_step_entropies[i]) / len(self._diag_step_entropies[i])
                if self._diag_step_entropies[i] else 0.0
                for i in range(steps)
            ]

            # Average final loss
            avg_final_loss = (
                sum(self._diag_final_losses) / len(self._diag_final_losses)
                if self._diag_final_losses else 0.0
            )

            # Compute max entropy for percentage calculation
            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            max_entropy = math.log(unwrapped_model.config.vocab_size)

            # Build log dict
            # Note: Use eval_ prefix (not eval/) so wandb's rewrite_logs converts to eval/
            all_logs = {}
            for i in range(10):
                all_logs[f"eval_loss_t_{i*10}-{(i+1)*10}pct"] = avg_bucket_losses[i]
            for i in range(steps):
                all_logs[f"eval_step_{i}_loss"] = avg_step_losses[i]
                all_logs[f"eval_step_{i}_geodist"] = avg_step_distances[i]
                all_logs[f"eval_step_{i}_tv"] = avg_step_tvs[i]
                all_logs[f"eval_step_{i}_entropy"] = avg_step_entropies[i]
                all_logs[f"eval_step_{i}_entropy_pct"] = 100.0 * avg_step_entropies[i] / max_entropy
            all_logs["eval_final_loss"] = avg_final_loss
            all_logs["eval_mean_step_tv"] = sum(avg_step_tvs) / len(avg_step_tvs) if avg_step_tvs else 0.0
            all_logs["eval_mean_step_entropy"] = sum(avg_step_entropies) / len(avg_step_entropies) if avg_step_entropies else 0.0
            all_logs["eval_mean_step_entropy_pct"] = 100.0 * all_logs["eval_mean_step_entropy"] / max_entropy

            # Log once per evaluation
            self.log(all_logs)

            # Print summary
            scaling_mode = "simple dt" if self.args.use_simple_dt else "endpoint (alpha_t_prime*dt/(1-alpha_t))"
            eval_temp = getattr(self.args, 'eval_temperature', 0.0)
            temp_str = f"{eval_temp}" if eval_temp > 0 else "none (1.0)"
            use_expected_logmap = getattr(self.args, 'use_expected_logmap', False)
            logmap_mode = "expected_logmap_to_onehots" if use_expected_logmap else "log_map(x_t, sqrt(probs))"
            print(f"\n=== Diagnostic Summary (averaged over {self._diag_batch_count} batches) ===")
            print(f"Integration scaling: {scaling_mode}")
            print(f"Eval temperature: {temp_str}")
            print(f"Log map mode: {logmap_mode}")
            print(f"Loss by time bucket (training-style):")
            for i in range(10):
                print(f"  t={i*10}-{(i+1)*10}%: {avg_bucket_losses[i]:.4f}")
            print(f"\nLoss by integration step (eval-style):")
            for i in range(0, steps, 4):
                entropy_pct = 100.0 * avg_step_entropies[i] / max_entropy
                print(f"  step {i}: loss={avg_step_losses[i]:.4f}, geodist={avg_step_distances[i]:.4f}, tv={avg_step_tvs[i]:.4f}, entropy={entropy_pct:.1f}%")
            print(f"\nFinal eval loss: {avg_final_loss:.4f}")
            print(f"Mean step TV: {sum(avg_step_tvs) / len(avg_step_tvs):.4f}")
            mean_entropy_pct = 100.0 * sum(avg_step_entropies) / len(avg_step_entropies) / max_entropy if avg_step_entropies else 0.0
            print(f"Mean step entropy: {mean_entropy_pct:.1f}% of max (low = confident, high = uncertain)")
            print("=" * 50)

        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Extended evaluation with diagnostic logging.
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        unwrapped_model = model.module if hasattr(model, "module") else model
        vocab_size = unwrapped_model.config.vocab_size
        device = input_ids.device

        loss_mask = labels != -100

        # Use unwrapped model to get embedding layer
        embed_layer = unwrapped_model.get_input_embeddings()

        # ============================================================
        # DIAGNOSTIC 1: Loss bucketed by time t (single-step, training-style)
        # ============================================================
        t_buckets = torch.linspace(0.001, 1.0, 11, device=device)  # 10 buckets
        bucket_losses = []

        for i in range(10):
            t_low, t_high = t_buckets[i], t_buckets[i + 1]
            t = t_low + (t_high - t_low) * torch.rand(b, device=device)

            # Sample noise
            x_0 = uniform_prior((b, l, vocab_size), device=device, dtype=embed_layer.weight.dtype)

            # Geodesic interpolation
            x_t = geodesic_interpolant_to_onehot(x_0, input_ids, t)

            # Keep prompt clean
            if not loss_mask.all():
                prompt_mask = ~loss_mask
                x_t[prompt_mask] = 0
                prompt_indices = input_ids[prompt_mask].unsqueeze(-1)
                x_t[prompt_mask] = x_t[prompt_mask].scatter(-1, prompt_indices, 1.0)

            # Forward pass
            x_embed = x_t if self.embed_type == "spherical" else sphere_to_simplex(x_t)
            soft_embeddings = torch.matmul(x_embed, embed_layer.weight)

            with torch.no_grad():
                outputs = model(inputs_embeds=soft_embeddings, attention_mask=attention_mask)
                logits = outputs.logits

                token_loss = F.cross_entropy(
                    logits.transpose(1, 2), input_ids, reduction="none"
                )
                masked_loss = (token_loss * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                bucket_losses.append(masked_loss.item())

        # Accumulate bucket losses (don't log per-batch)
        for i in range(10):
            self._diag_bucket_losses[i].append(bucket_losses[i])

        # ============================================================
        # DIAGNOSTIC 2: Loss at each integration step (multi-step, eval-style)
        # ============================================================
        prompt_onehot = F.one_hot(input_ids, num_classes=vocab_size).float()
        x_sphere = simplex_to_sphere(prompt_onehot)

        prior_sample = uniform_prior((b, l, vocab_size), device=device)
        x_sphere = torch.where(
            loss_mask.unsqueeze(-1).expand_as(x_sphere),
            prior_sample,
            x_sphere,
        )

        context_embeds = embed_layer(input_ids)

        # Manual flow integration with per-step logging
        steps = self.eval_integration_steps
        timesteps = torch.linspace(0, 1, steps + 1, device=device)
        step_losses = []
        step_distances = []
        step_tvs = []
        step_entropies = []

        # Pre-expand flow_mask
        flow_mask_expanded = loss_mask.unsqueeze(-1).expand_as(x_sphere)

        for step_idx in range(steps):
            t_curr = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            dt = t_next - t_curr

            # Get step weight
            if self.args.use_simple_dt:
                # Simple dt scaling like fisher-flow's tangent_euler
                step_weight = dt
            else:
                # Endpoint prediction formula: alpha_t_prime * dt / (1 - alpha_t)
                if self.schedule_type == "linear":
                    alpha_t = t_curr
                    alpha_t_prime = torch.ones_like(t_curr)
                else:
                    # Cosine schedule
                    import math
                    alpha_t = 1 - torch.cos(math.pi / 2 * t_curr).square()
                    alpha_t_prime = math.pi / 2 * torch.sin(math.pi * t_curr)
                step_weight = (alpha_t_prime * dt / (1 - alpha_t + 1e-5))

            # Compute soft embeddings
            x_embed = x_sphere if self.embed_type == "spherical" else sphere_to_simplex(x_sphere)
            soft_embeddings = torch.matmul(x_embed.to(embed_layer.weight.dtype), embed_layer.weight)
            soft_embeddings = torch.where(
                loss_mask.unsqueeze(-1).expand_as(soft_embeddings),
                soft_embeddings,
                context_embeds,
            )

            with torch.no_grad():
                # Save x_sphere before step for TV computation
                x_sphere_before_step = x_sphere.clone()

                outputs = model(inputs_embeds=soft_embeddings, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss at this step (how well does model predict target?)
                step_ce = F.cross_entropy(
                    logits.transpose(1, 2), input_ids, reduction="none"
                )
                masked_step_loss = (step_ce * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_losses.append(masked_step_loss.item())

                # ============================================================
                # DIAGNOSTIC 3: Compare x_sphere to ground-truth x_t at this timestep
                # ============================================================
                # Ground truth: where should x be at time t_next?
                x_0_gt = prior_sample  # Same noise we started with
                x_t_gt = geodesic_interpolant_to_onehot(x_0_gt, input_ids, t_next.unsqueeze(0).expand(b))

                # Apply temperature scaling for integration (logits / temperature)
                # temperature < 1.0 sharpens predictions, > 1.0 softens them
                eval_temp = getattr(self.args, 'eval_temperature', 0.0)
                if eval_temp > 0:
                    logits = logits / eval_temp

                # Take integration step - different for CE vs MSE
                if self.loss_type == "mse":
                    # MSE/Velocity prediction: logits are velocity, project to tangent space
                    velocity = make_tangent(x_sphere, logits)
                    # For velocity prediction, use simple dt scaling
                    tangent = velocity * dt
                else:
                    # CE/Endpoint prediction: logits -> probs -> sphere point
                    probs = F.softmax(logits, dim=-1)

                    # Check if we should use expected_logmap_to_onehots
                    use_expected_logmap = getattr(self.args, 'use_expected_logmap', False)
                    if use_expected_logmap:
                        # Use expected_logmap_to_onehots: mathematically correct expected direction
                        tangent = expected_logmap_to_onehots(x_sphere, probs) * step_weight
                    else:
                        # Standard approach: log_map(x_t, sqrt(probs))
                        x_1_pred = probs.sqrt()
                        # Log map for geodesic step
                        dot_pq = (x_sphere * x_1_pred).sum(dim=-1, keepdim=True)
                        q_proj = x_1_pred - dot_pq * x_sphere
                        q_proj_norm = torch.norm(q_proj, dim=-1, keepdim=True).clamp(min=1e-8)
                        dot_clamped = dot_pq.clamp(-1 + 1e-7, 1 - 1e-7)
                        dist = torch.acos(dot_clamped)
                        tangent = q_proj / q_proj_norm * dist * step_weight

                # Apply exp_map to get new position
                v_norm = torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)
                x_sphere_new = x_sphere * torch.cos(v_norm) + tangent * torch.sin(v_norm) / v_norm
                x_sphere_new = x_sphere_new / torch.norm(x_sphere_new, dim=-1, keepdim=True).clamp(min=1e-8)

                # Update only flow positions
                x_sphere = torch.where(flow_mask_expanded, x_sphere_new, x_sphere)

                # Compute geodesic distance between model's x_sphere and ground-truth x_t
                # Only for flow positions
                dot_model_gt = (x_sphere * x_t_gt).sum(dim=-1)  # [b, l]
                geodesic_dist = torch.acos(dot_model_gt.clamp(-1 + 1e-7, 1 - 1e-7))  # [b, l]
                mean_dist = (geodesic_dist * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_distances.append(mean_dist.item())

                # Compute total variation between model's predicted distribution and input
                # TV = 0.5 * sum(|p_pred - p_input|) averaged over flow positions
                # p_input is the distribution BEFORE the step (x_sphere before update)
                # p_pred is the model's endpoint prediction (softmax of logits)
                p_input = sphere_to_simplex(x_sphere_before_step)
                p_pred = F.softmax(logits, dim=-1)
                tv_per_pos = 0.5 * (p_pred - p_input).abs().sum(dim=-1)  # [b, l]
                mean_tv = (tv_per_pos * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_tvs.append(mean_tv.item())

                # Compute entropy of predicted distribution
                # H(p) = -sum(p * log(p)), measures model uncertainty
                # Low entropy = confident (peaked), high entropy = uncertain (spread out)
                # This helps diagnose if expected_logmap_to_onehots would help
                entropy_per_pos = -(p_pred * p_pred.clamp(min=1e-8).log()).sum(dim=-1)  # [b, l]
                mean_entropy = (entropy_per_pos * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_entropies.append(mean_entropy.item())

        # Accumulate step losses, distances, TVs, and entropies (don't log per-batch)
        for i in range(steps):
            self._diag_step_losses[i].append(step_losses[i])
            self._diag_step_distances[i].append(step_distances[i])
            self._diag_step_tvs[i].append(step_tvs[i])
            self._diag_step_entropies[i].append(step_entropies[i])

        # Final eval (standard)
        final_probs = sphere_to_simplex(x_sphere)
        final_log_probs = torch.log(final_probs.clamp(min=1e-10))

        # Use nll_loss since we already have log probs (not cross_entropy which applies log_softmax)
        token_nll = F.nll_loss(
            final_log_probs.transpose(1, 2), input_ids, reduction="none"
        )
        token_nll = token_nll * loss_mask.float()

        self.meter.update(
            split="eval",
            value=token_nll.detach(),
            weight=loss_mask.float().detach(),
        )

        loss = token_nll.sum() / loss_mask.sum().clamp_min(1)

        # Accumulate final loss and increment batch count
        self._diag_final_losses.append(loss.item())
        self._diag_batch_count += 1

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), logits.detach().contiguous(), labels.detach().contiguous())


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        if not data_args.skip_post_process:
            dataset = dllm.utils.post_process_dataset(dataset, data_args)

    if training_args.group_by_length and "length" in dataset["train"].column_names:
        from datasets.arrow_dataset import Column
        from transformers.trainer_pt_utils import LengthGroupedSampler
        _original_init = LengthGroupedSampler.__init__

        def _patched_init(self, batch_size, *, dataset=None, lengths=None, **kwargs):
            if lengths is not None and not isinstance(lengths, list):
                if isinstance(lengths, Column):
                    lengths = dataset.data[lengths.column_name].to_pylist()
                elif hasattr(lengths, "to_pylist"):
                    lengths = lengths.to_pylist()
                else:
                    lengths = list(lengths)
            _original_init(self, batch_size, dataset=dataset, lengths=lengths, **kwargs)

        LengthGroupedSampler.__init__ = _patched_init

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start diagnostic training...")

    trainer = DiagnosticBertSFMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,
                ),
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
