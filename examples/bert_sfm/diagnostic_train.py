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
    # Significantly more stable and accurate than the legacy sqrt(probs) approach.
    # Default: True (recommended). Set to False only for legacy compatibility.
    use_expected_logmap: bool = True


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
        # Local contraction test accumulators (per step)
        self._diag_contraction_ratios = [[] for _ in range(steps)]

        # NEW DIAGNOSTICS
        # 1) One-step improvement test: ΔL = L_{k+1} - L_k bucketed by entropy
        #    3 entropy buckets: low, medium, high
        self._diag_delta_L_low_entropy = [[] for _ in range(steps)]
        self._diag_delta_L_med_entropy = [[] for _ in range(steps)]
        self._diag_delta_L_high_entropy = [[] for _ in range(steps)]

        # 2) Directional contraction: along model's update direction (u) and GT direction (g)
        self._diag_contraction_u = [[] for _ in range(steps)]  # Along update tangent
        self._diag_contraction_g = [[] for _ in range(steps)]  # Along GT tangent

        # 3) Support-violation / OOD distance from training manifold
        self._diag_ood_distance = [[] for _ in range(steps)]

        # 4) Calibration: accuracy and confidence for ECE computation
        self._diag_calibration_correct = [[] for _ in range(10)]  # 10 confidence buckets
        self._diag_calibration_confidence = [[] for _ in range(10)]
        self._diag_calibration_count = [[] for _ in range(10)]

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

            # Average contraction ratios
            avg_contraction_ratios = [
                sum(self._diag_contraction_ratios[i]) / len(self._diag_contraction_ratios[i])
                if self._diag_contraction_ratios[i] else 0.0
                for i in range(steps)
            ]

            # NEW: Average ΔL by entropy bucket
            avg_delta_L_low = [
                sum(self._diag_delta_L_low_entropy[i]) / len(self._diag_delta_L_low_entropy[i])
                if self._diag_delta_L_low_entropy[i] else 0.0
                for i in range(steps)
            ]
            avg_delta_L_med = [
                sum(self._diag_delta_L_med_entropy[i]) / len(self._diag_delta_L_med_entropy[i])
                if self._diag_delta_L_med_entropy[i] else 0.0
                for i in range(steps)
            ]
            avg_delta_L_high = [
                sum(self._diag_delta_L_high_entropy[i]) / len(self._diag_delta_L_high_entropy[i])
                if self._diag_delta_L_high_entropy[i] else 0.0
                for i in range(steps)
            ]

            # NEW: Average directional contraction ratios
            avg_contraction_u = [
                sum(self._diag_contraction_u[i]) / len(self._diag_contraction_u[i])
                if self._diag_contraction_u[i] else 0.0
                for i in range(steps)
            ]
            avg_contraction_g = [
                sum(self._diag_contraction_g[i]) / len(self._diag_contraction_g[i])
                if self._diag_contraction_g[i] else 0.0
                for i in range(steps)
            ]

            # NEW: Average OOD distance
            avg_ood_distance = [
                sum(self._diag_ood_distance[i]) / len(self._diag_ood_distance[i])
                if self._diag_ood_distance[i] else 0.0
                for i in range(steps)
            ]

            # NEW: Compute ECE (Expected Calibration Error) from calibration buckets
            ece = 0.0
            total_samples = 0
            calibration_gaps = []
            for bucket_idx in range(10):
                if self._diag_calibration_count[bucket_idx]:
                    bucket_count = sum(self._diag_calibration_count[bucket_idx])
                    if bucket_count > 0:
                        # Weighted average of accuracy and confidence for this bucket
                        bucket_correct = sum(
                            c * n for c, n in zip(self._diag_calibration_correct[bucket_idx], self._diag_calibration_count[bucket_idx])
                        ) / bucket_count
                        bucket_conf = sum(
                            c * n for c, n in zip(self._diag_calibration_confidence[bucket_idx], self._diag_calibration_count[bucket_idx])
                        ) / bucket_count
                        gap = abs(bucket_correct - bucket_conf)
                        calibration_gaps.append((bucket_idx, bucket_correct, bucket_conf, gap, bucket_count))
                        ece += gap * bucket_count
                        total_samples += bucket_count
            if total_samples > 0:
                ece /= total_samples

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
            # Log contraction ratios per step
            for i in range(steps):
                all_logs[f"eval_step_{i}_contraction"] = avg_contraction_ratios[i]
            all_logs["eval_mean_contraction"] = sum(avg_contraction_ratios) / len(avg_contraction_ratios) if avg_contraction_ratios else 0.0

            # NEW: Log ΔL by entropy bucket
            for i in range(steps):
                all_logs[f"eval_step_{i}_delta_L_low_entropy"] = avg_delta_L_low[i]
                all_logs[f"eval_step_{i}_delta_L_med_entropy"] = avg_delta_L_med[i]
                all_logs[f"eval_step_{i}_delta_L_high_entropy"] = avg_delta_L_high[i]
            all_logs["eval_mean_delta_L_low_entropy"] = sum(avg_delta_L_low) / len(avg_delta_L_low) if avg_delta_L_low else 0.0
            all_logs["eval_mean_delta_L_med_entropy"] = sum(avg_delta_L_med) / len(avg_delta_L_med) if avg_delta_L_med else 0.0
            all_logs["eval_mean_delta_L_high_entropy"] = sum(avg_delta_L_high) / len(avg_delta_L_high) if avg_delta_L_high else 0.0

            # NEW: Log directional contraction ratios
            for i in range(steps):
                all_logs[f"eval_step_{i}_contraction_u"] = avg_contraction_u[i]
                all_logs[f"eval_step_{i}_contraction_g"] = avg_contraction_g[i]
            all_logs["eval_mean_contraction_u"] = sum(avg_contraction_u) / len(avg_contraction_u) if avg_contraction_u else 0.0
            all_logs["eval_mean_contraction_g"] = sum(avg_contraction_g) / len(avg_contraction_g) if avg_contraction_g else 0.0

            # NEW: Log OOD distance
            for i in range(steps):
                all_logs[f"eval_step_{i}_ood_distance"] = avg_ood_distance[i]
            all_logs["eval_mean_ood_distance"] = sum(avg_ood_distance) / len(avg_ood_distance) if avg_ood_distance else 0.0

            # NEW: Log ECE (Expected Calibration Error)
            all_logs["eval_ece"] = ece

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
            mean_contraction = sum(avg_contraction_ratios) / len(avg_contraction_ratios) if avg_contraction_ratios else 0.0
            print(f"\nLocal contraction test (rho = d(Phi(x'), Phi(x)) / d(x', x)):")
            print(f"  Mean contraction ratio: {mean_contraction:.4f}")
            if mean_contraction < 1.0:
                print(f"  -> Locally CONTRACTIVE (rho < 1): multi-step should help")
            else:
                print(f"  -> Locally EXPANSIVE (rho > 1): multi-step may diverge")
            print(f"  Per-step contraction ratios:")
            for i in range(0, steps, 4):
                print(f"    step {i}: rho={avg_contraction_ratios[i]:.4f}")

            # NEW: One-step improvement test (ΔL by entropy)
            mean_delta_L_low = sum(avg_delta_L_low) / len(avg_delta_L_low) if avg_delta_L_low else 0.0
            mean_delta_L_med = sum(avg_delta_L_med) / len(avg_delta_L_med) if avg_delta_L_med else 0.0
            mean_delta_L_high = sum(avg_delta_L_high) / len(avg_delta_L_high) if avg_delta_L_high else 0.0
            print(f"\nOne-step improvement test (ΔL = L_after - L_before, by entropy bucket):")
            print(f"  Low entropy (confident):  ΔL = {mean_delta_L_low:+.4f}")
            print(f"  Med entropy (moderate):   ΔL = {mean_delta_L_med:+.4f}")
            print(f"  High entropy (uncertain): ΔL = {mean_delta_L_high:+.4f}")
            if mean_delta_L_low > 0.1 and mean_delta_L_low > mean_delta_L_high:
                print(f"  -> WARNING: Low entropy positions have POSITIVE ΔL (confirmation bias!)")
            print(f"  Per-step ΔL (low entropy):")
            for i in range(0, steps, 4):
                print(f"    step {i}: ΔL_low={avg_delta_L_low[i]:+.4f}, ΔL_med={avg_delta_L_med[i]:+.4f}, ΔL_high={avg_delta_L_high[i]:+.4f}")

            # NEW: Directional contraction test
            mean_contraction_u = sum(avg_contraction_u) / len(avg_contraction_u) if avg_contraction_u else 0.0
            mean_contraction_g = sum(avg_contraction_g) / len(avg_contraction_g) if avg_contraction_g else 0.0
            print(f"\nDirectional contraction test:")
            print(f"  Along update direction (u): rho_u = {mean_contraction_u:.4f}")
            print(f"  Along GT direction (g):     rho_g = {mean_contraction_g:.4f}")
            if mean_contraction_u > 1.0:
                print(f"  -> WARNING: Expansive along model's update direction (instability in own mistakes)")
            print(f"  Per-step directional contraction:")
            for i in range(0, steps, 4):
                print(f"    step {i}: rho_u={avg_contraction_u[i]:.4f}, rho_g={avg_contraction_g[i]:.4f}")

            # NEW: OOD distance from training manifold
            mean_ood_dist = sum(avg_ood_distance) / len(avg_ood_distance) if avg_ood_distance else 0.0
            print(f"\nOOD distance from training manifold (lower = closer to training distribution):")
            print(f"  Mean OOD distance: {mean_ood_dist:.4f}")
            print(f"  Per-step OOD distance:")
            for i in range(0, steps, 4):
                print(f"    step {i}: d_ood={avg_ood_distance[i]:.4f}")

            # NEW: Calibration / ECE
            print(f"\nCalibration (Expected Calibration Error):")
            print(f"  ECE = {ece:.4f} (lower = better calibrated)")
            if calibration_gaps:
                print(f"  Confidence buckets (accuracy vs confidence):")
                for bucket_idx, acc, conf, gap, count in calibration_gaps:
                    bucket_range = f"{bucket_idx*10}-{(bucket_idx+1)*10}%"
                    overconf = "overconfident" if conf > acc else "underconfident"
                    print(f"    {bucket_range}: acc={acc:.3f}, conf={conf:.3f}, gap={gap:.3f} ({overconf}, n={int(count)})")

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
        step_contractions = []

        # NEW DIAGNOSTIC ACCUMULATORS (per-batch)
        step_delta_L_low = []
        step_delta_L_med = []
        step_delta_L_high = []
        step_contraction_u = []
        step_contraction_g = []
        step_ood_distance = []
        prev_loss = None  # For computing ΔL

        # Pre-expand flow_mask
        flow_mask_expanded = loss_mask.unsqueeze(-1).expand_as(x_sphere)

        # ============================================================
        # DIAGNOSTIC 4: Local contraction test setup
        # Create a perturbed version of x_sphere to test if sampler is contractive
        # x' = exp_x(epsilon * r) where r is a random tangent vector
        # ============================================================
        epsilon = 0.01  # Small perturbation magnitude
        # Generate random tangent vector at x_sphere
        random_vec = torch.randn_like(x_sphere)
        # Project to tangent space: r = v - <v, x> * x
        dot_xr = (x_sphere * random_vec).sum(dim=-1, keepdim=True)
        tangent_r = random_vec - dot_xr * x_sphere
        # Normalize and scale by epsilon
        tangent_r = tangent_r / torch.norm(tangent_r, dim=-1, keepdim=True).clamp(min=1e-8) * epsilon
        # Apply exp_map to get perturbed point x'
        x_sphere_perturbed = exp_map(x_sphere, tangent_r)
        # Compute initial distance d(x, x') for normalization
        dot_xx_prime = (x_sphere * x_sphere_perturbed).sum(dim=-1)
        initial_dist = torch.acos(dot_xx_prime.clamp(-1 + 1e-7, 1 - 1e-7))  # [b, l]

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
                    use_expected_logmap = getattr(self.args, 'use_expected_logmap', True)
                    if use_expected_logmap:
                        # Use expected_logmap_to_onehots: mathematically correct expected direction
                        tangent = expected_logmap_to_onehots(x_sphere, probs) * step_weight
                    else:
                        # DEPRECATED: Legacy approach using log_map(x_t, sqrt(probs))
                        # This approximation introduces systematic bias when probs is not sharply peaked.
                        # Use use_expected_logmap=True (default) for better stability and accuracy.
                        x_1_pred = probs.sqrt()
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

                # ============================================================
                # DIAGNOSTIC 4: Local contraction test
                # Run the same step on the perturbed x_sphere_perturbed and measure
                # rho = d(Phi(x), Phi(x')) / d(x, x')
                # If rho < 1 -> locally contractive (multi-step helps)
                # If rho > 1 -> locally expansive (multi-step may diverge)
                # ============================================================
                # Compute soft embeddings for perturbed state
                x_embed_p = x_sphere_perturbed if self.embed_type == "spherical" else sphere_to_simplex(x_sphere_perturbed)
                soft_embeddings_p = torch.matmul(x_embed_p.to(embed_layer.weight.dtype), embed_layer.weight)
                soft_embeddings_p = torch.where(
                    loss_mask.unsqueeze(-1).expand_as(soft_embeddings_p),
                    soft_embeddings_p,
                    context_embeds,
                )

                # Forward pass on perturbed state
                outputs_p = model(inputs_embeds=soft_embeddings_p, attention_mask=attention_mask)
                logits_p = outputs_p.logits

                # Apply same temperature
                if eval_temp > 0:
                    logits_p = logits_p / eval_temp

                # Take integration step on perturbed state (same as main state)
                if self.loss_type == "mse":
                    velocity_p = make_tangent(x_sphere_perturbed, logits_p)
                    tangent_p = velocity_p * dt
                else:
                    probs_p = F.softmax(logits_p, dim=-1)
                    if use_expected_logmap:
                        tangent_p = expected_logmap_to_onehots(x_sphere_perturbed, probs_p) * step_weight
                    else:
                        x_1_pred_p = probs_p.sqrt()
                        dot_pq_p = (x_sphere_perturbed * x_1_pred_p).sum(dim=-1, keepdim=True)
                        q_proj_p = x_1_pred_p - dot_pq_p * x_sphere_perturbed
                        q_proj_norm_p = torch.norm(q_proj_p, dim=-1, keepdim=True).clamp(min=1e-8)
                        dot_clamped_p = dot_pq_p.clamp(-1 + 1e-7, 1 - 1e-7)
                        dist_p = torch.acos(dot_clamped_p)
                        tangent_p = q_proj_p / q_proj_norm_p * dist_p * step_weight

                # Apply exp_map to get new perturbed position
                v_norm_p = torch.norm(tangent_p, dim=-1, keepdim=True).clamp(min=1e-8)
                x_sphere_perturbed_new = x_sphere_perturbed * torch.cos(v_norm_p) + tangent_p * torch.sin(v_norm_p) / v_norm_p
                x_sphere_perturbed_new = x_sphere_perturbed_new / torch.norm(x_sphere_perturbed_new, dim=-1, keepdim=True).clamp(min=1e-8)

                # Update only flow positions for perturbed state
                x_sphere_perturbed = torch.where(flow_mask_expanded, x_sphere_perturbed_new, x_sphere_perturbed)

                # Compute distance after step: d(Phi(x), Phi(x'))
                dot_after = (x_sphere * x_sphere_perturbed).sum(dim=-1)
                dist_after = torch.acos(dot_after.clamp(-1 + 1e-7, 1 - 1e-7))  # [b, l]

                # Compute contraction ratio rho = d_after / d_before
                # Use initial_dist (before any steps) as d_before to avoid division issues
                # For per-step ratio, we should track d_before at each step
                contraction_ratio = dist_after / initial_dist.clamp(min=1e-8)
                mean_contraction = (contraction_ratio * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_contractions.append(mean_contraction.item())

                # Update initial_dist for next step (so we measure per-step contraction)
                initial_dist = dist_after

                # ============================================================
                # NEW DIAGNOSTIC 1: One-step improvement test (ΔL bucketed by entropy)
                # Measures ΔL = L_{k+1} - L_k and buckets by model's entropy at step k
                # If low entropy + wrong → ΔL strongly positive (catastrophic feedback)
                # ============================================================
                # We already have step_ce (CE loss at this step) and entropy_per_pos
                # Compute loss AFTER the step (L_{k+1}) by running model on updated x_sphere
                x_embed_after = x_sphere if self.embed_type == "spherical" else sphere_to_simplex(x_sphere)
                soft_embeddings_after = torch.matmul(x_embed_after.to(embed_layer.weight.dtype), embed_layer.weight)
                soft_embeddings_after = torch.where(
                    loss_mask.unsqueeze(-1).expand_as(soft_embeddings_after),
                    soft_embeddings_after,
                    context_embeds,
                )
                outputs_after = model(inputs_embeds=soft_embeddings_after, attention_mask=attention_mask)
                logits_after = outputs_after.logits
                step_ce_after = F.cross_entropy(
                    logits_after.transpose(1, 2), input_ids, reduction="none"
                )  # [b, l]

                # ΔL per position = L_after - L_before
                delta_L = step_ce_after - step_ce  # [b, l]

                # Bucket by entropy (using entropy from BEFORE the step)
                # Entropy thresholds: low < 2.0, medium < 5.0, high >= 5.0 (in nats)
                unwrapped = model.module if hasattr(model, "module") else model
                max_ent = math.log(unwrapped.config.vocab_size)
                low_thresh = 0.2 * max_ent  # ~20% of max entropy
                high_thresh = 0.6 * max_ent  # ~60% of max entropy

                low_entropy_mask = (entropy_per_pos < low_thresh) & loss_mask
                med_entropy_mask = (entropy_per_pos >= low_thresh) & (entropy_per_pos < high_thresh) & loss_mask
                high_entropy_mask = (entropy_per_pos >= high_thresh) & loss_mask

                # Compute mean ΔL for each bucket
                if low_entropy_mask.sum() > 0:
                    delta_L_low = (delta_L * low_entropy_mask.float()).sum() / low_entropy_mask.sum()
                    step_delta_L_low.append(delta_L_low.item())
                else:
                    step_delta_L_low.append(0.0)

                if med_entropy_mask.sum() > 0:
                    delta_L_med = (delta_L * med_entropy_mask.float()).sum() / med_entropy_mask.sum()
                    step_delta_L_med.append(delta_L_med.item())
                else:
                    step_delta_L_med.append(0.0)

                if high_entropy_mask.sum() > 0:
                    delta_L_high = (delta_L * high_entropy_mask.float()).sum() / high_entropy_mask.sum()
                    step_delta_L_high.append(delta_L_high.item())
                else:
                    step_delta_L_high.append(0.0)

                # ============================================================
                # NEW DIAGNOSTIC 2: Directional contraction test
                # Measure contraction along the model's update direction (u) and GT direction (g)
                # u_k = update tangent (expected one-hot drift)
                # g_k = log_map(x_k, e_y) = direction to ground truth
                # ============================================================
                epsilon_dir = 0.01  # Small perturbation magnitude

                # u_k = the tangent we computed (normalized)
                u_k = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)

                # g_k = log_map(x_sphere_before_step, one_hot(input_ids))
                # Direction to ground truth one-hot
                gt_onehot = F.one_hot(input_ids, num_classes=vocab_size).float()
                gt_sphere = simplex_to_sphere(gt_onehot)  # sqrt of one-hot = one-hot on sphere
                g_k = log_map(x_sphere_before_step, gt_sphere)
                g_k = g_k / torch.norm(g_k, dim=-1, keepdim=True).clamp(min=1e-8)

                # Create perturbed states along u and g directions
                x_perturb_u = exp_map(x_sphere_before_step, epsilon_dir * u_k)
                x_perturb_g = exp_map(x_sphere_before_step, epsilon_dir * g_k)

                # Run one step on each perturbed state
                # For x_perturb_u:
                x_embed_u = x_perturb_u if self.embed_type == "spherical" else sphere_to_simplex(x_perturb_u)
                soft_emb_u = torch.matmul(x_embed_u.to(embed_layer.weight.dtype), embed_layer.weight)
                soft_emb_u = torch.where(loss_mask.unsqueeze(-1).expand_as(soft_emb_u), soft_emb_u, context_embeds)
                outputs_u = model(inputs_embeds=soft_emb_u, attention_mask=attention_mask)
                logits_u = outputs_u.logits
                if eval_temp > 0:
                    logits_u = logits_u / eval_temp
                probs_u = F.softmax(logits_u, dim=-1)
                if use_expected_logmap:
                    tangent_u = expected_logmap_to_onehots(x_perturb_u, probs_u) * step_weight
                else:
                    x1_u = probs_u.sqrt()
                    tangent_u = log_map(x_perturb_u, x1_u) * step_weight
                v_norm_u = torch.norm(tangent_u, dim=-1, keepdim=True).clamp(min=1e-8)
                x_after_u = x_perturb_u * torch.cos(v_norm_u) + tangent_u * torch.sin(v_norm_u) / v_norm_u
                x_after_u = x_after_u / torch.norm(x_after_u, dim=-1, keepdim=True).clamp(min=1e-8)

                # For x_perturb_g:
                x_embed_g = x_perturb_g if self.embed_type == "spherical" else sphere_to_simplex(x_perturb_g)
                soft_emb_g = torch.matmul(x_embed_g.to(embed_layer.weight.dtype), embed_layer.weight)
                soft_emb_g = torch.where(loss_mask.unsqueeze(-1).expand_as(soft_emb_g), soft_emb_g, context_embeds)
                outputs_g = model(inputs_embeds=soft_emb_g, attention_mask=attention_mask)
                logits_g = outputs_g.logits
                if eval_temp > 0:
                    logits_g = logits_g / eval_temp
                probs_g = F.softmax(logits_g, dim=-1)
                if use_expected_logmap:
                    tangent_g = expected_logmap_to_onehots(x_perturb_g, probs_g) * step_weight
                else:
                    x1_g = probs_g.sqrt()
                    tangent_g = log_map(x_perturb_g, x1_g) * step_weight
                v_norm_g = torch.norm(tangent_g, dim=-1, keepdim=True).clamp(min=1e-8)
                x_after_g = x_perturb_g * torch.cos(v_norm_g) + tangent_g * torch.sin(v_norm_g) / v_norm_g
                x_after_g = x_after_g / torch.norm(x_after_g, dim=-1, keepdim=True).clamp(min=1e-8)

                # Compute distances
                # d(x, x_perturb_u) before step
                d_before_u = torch.acos((x_sphere_before_step * x_perturb_u).sum(dim=-1).clamp(-1+1e-7, 1-1e-7))
                # d(Phi(x), Phi(x_perturb_u)) after step
                d_after_u = torch.acos((x_sphere * x_after_u).sum(dim=-1).clamp(-1+1e-7, 1-1e-7))
                rho_u = d_after_u / d_before_u.clamp(min=1e-8)
                mean_rho_u = (rho_u * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_contraction_u.append(mean_rho_u.item())

                # Same for g direction
                d_before_g = torch.acos((x_sphere_before_step * x_perturb_g).sum(dim=-1).clamp(-1+1e-7, 1-1e-7))
                d_after_g = torch.acos((x_sphere * x_after_g).sum(dim=-1).clamp(-1+1e-7, 1-1e-7))
                rho_g = d_after_g / d_before_g.clamp(min=1e-8)
                mean_rho_g = (rho_g * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_contraction_g.append(mean_rho_g.item())

                # ============================================================
                # NEW DIAGNOSTIC 3: Support-violation / OOD distance
                # Measure how far x_sphere is from the training manifold M_t
                # M_t = {geodesic(x_0, e_k, alpha(t))} for k in top-K tokens
                # residual r = min_k d(x, m_{t,k})
                # ============================================================
                # Get alpha(t) for current time
                if self.schedule_type == "linear":
                    alpha_t_curr = t_next
                else:
                    alpha_t_curr = 1 - torch.cos(math.pi / 2 * t_next).square()

                # Get top-K tokens under the model (K=5)
                top_k = 5
                _, top_k_indices = torch.topk(probs, top_k, dim=-1)  # [b, l, K]

                # For each position, compute distance to closest training manifold point
                # m_{t,k} = geodesic(x_0, e_k, alpha(t)) where x_0 is the prior sample
                min_distances = torch.full((b, l), float('inf'), device=device)

                for k_idx in range(top_k):
                    # Get the k-th top token indices
                    token_k = top_k_indices[:, :, k_idx]  # [b, l]
                    # Create one-hot for this token
                    onehot_k = F.one_hot(token_k, num_classes=vocab_size).float()
                    sphere_k = simplex_to_sphere(onehot_k)
                    # Compute training manifold point: geodesic(prior_sample, sphere_k, alpha_t)
                    # Use geodesic_interpolant
                    m_t_k = geodesic_interpolant(prior_sample, sphere_k, alpha_t_curr)
                    # Distance from x_sphere to m_t_k
                    dot_xm = (x_sphere * m_t_k).sum(dim=-1)
                    dist_to_m = torch.acos(dot_xm.clamp(-1+1e-7, 1-1e-7))
                    min_distances = torch.minimum(min_distances, dist_to_m)

                mean_ood_dist = (min_distances * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_ood_distance.append(mean_ood_dist.item())

        # ============================================================
        # NEW DIAGNOSTIC 4: Calibration (compute once at end, not per-step)
        # Measure accuracy vs confidence for ECE computation
        # ============================================================
        # Use final x_sphere to get model's confidence
        final_probs_calib = sphere_to_simplex(x_sphere)
        final_confidence, final_predictions = final_probs_calib.max(dim=-1)  # [b, l]
        correct = (final_predictions == input_ids).float()  # [b, l]

        # Bucket by confidence (10 buckets: 0-0.1, 0.1-0.2, ..., 0.9-1.0)
        for bucket_idx in range(10):
            low_conf = bucket_idx / 10.0
            high_conf = (bucket_idx + 1) / 10.0
            bucket_mask = (final_confidence >= low_conf) & (final_confidence < high_conf) & loss_mask
            if bucket_mask.sum() > 0:
                bucket_correct = (correct * bucket_mask.float()).sum() / bucket_mask.sum()
                bucket_confidence = (final_confidence * bucket_mask.float()).sum() / bucket_mask.sum()
                self._diag_calibration_correct[bucket_idx].append(bucket_correct.item())
                self._diag_calibration_confidence[bucket_idx].append(bucket_confidence.item())
                self._diag_calibration_count[bucket_idx].append(bucket_mask.sum().item())

        # Accumulate step losses, distances, TVs, entropies, and contraction ratios (don't log per-batch)
        for i in range(steps):
            self._diag_step_losses[i].append(step_losses[i])
            self._diag_step_distances[i].append(step_distances[i])
            self._diag_step_tvs[i].append(step_tvs[i])
            self._diag_step_entropies[i].append(step_entropies[i])
            self._diag_contraction_ratios[i].append(step_contractions[i])
            # New diagnostics
            self._diag_delta_L_low_entropy[i].append(step_delta_L_low[i])
            self._diag_delta_L_med_entropy[i].append(step_delta_L_med[i])
            self._diag_delta_L_high_entropy[i].append(step_delta_L_high[i])
            self._diag_contraction_u[i].append(step_contraction_u[i])
            self._diag_contraction_g[i].append(step_contraction_g[i])
            self._diag_ood_distance[i].append(step_ood_distance[i])

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
