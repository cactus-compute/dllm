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
)
from dllm.pipelines.bert_sfm.trainer import geodesic_interpolant_to_onehot

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "prajjwal1/bert-medium"


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


class DiagnosticBertSFMTrainer(BertSFMTrainer):
    """Extended trainer with diagnostic logging during evaluation."""

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

        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        else:
            embed_layer = model.model.embed_tokens

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

        # Log bucket losses
        if hasattr(self, "_diagnostic_step"):
            self._diagnostic_step += 1
        else:
            self._diagnostic_step = 0

        bucket_log = {f"diag/loss_t_{i*10}-{(i+1)*10}pct": bucket_losses[i] for i in range(10)}

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
        steps = 20
        timesteps = torch.linspace(0, 1, steps + 1, device=device)
        step_losses = []
        step_distances = []

        # Pre-expand flow_mask
        flow_mask_expanded = loss_mask.unsqueeze(-1).expand_as(x_sphere)

        for step_idx in range(steps):
            t_curr = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            dt = t_next - t_curr

            # Get schedule
            if self.schedule_type == "linear":
                alpha_t = t_curr
                alpha_t_prime = torch.ones_like(t_curr)
            else:
                # Cosine schedule
                import math
                alpha_t = 1 - torch.cos(math.pi / 2 * t_curr).square()
                alpha_t_prime = math.pi / 2 * torch.sin(math.pi * t_curr)

            # Compute soft embeddings
            x_embed = x_sphere if self.embed_type == "spherical" else sphere_to_simplex(x_sphere)
            soft_embeddings = torch.matmul(x_embed.to(embed_layer.weight.dtype), embed_layer.weight)
            soft_embeddings = torch.where(
                loss_mask.unsqueeze(-1).expand_as(soft_embeddings),
                soft_embeddings,
                context_embeds,
            )

            with torch.no_grad():
                outputs = model(inputs_embeds=soft_embeddings, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute loss at this step (how well does model predict target?)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs.clamp(min=1e-10))
                step_ce = F.cross_entropy(
                    log_probs.transpose(1, 2), input_ids, reduction="none"
                )
                masked_step_loss = (step_ce * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
                step_losses.append(masked_step_loss.item())

                # ============================================================
                # DIAGNOSTIC 3: Compare x_sphere to ground-truth x_t at this timestep
                # ============================================================
                # Ground truth: where should x be at time t_next?
                x_0_gt = prior_sample  # Same noise we started with
                x_t_gt = geodesic_interpolant_to_onehot(x_0_gt, input_ids, t_next.unsqueeze(0).expand(b))

                # Take integration step
                x_1_pred = probs.sqrt()
                step_weight = (alpha_t_prime * dt / (1 - alpha_t + 1e-5))

                # Log map and exp map for geodesic step
                dot_pq = (x_sphere * x_1_pred).sum(dim=-1, keepdim=True)
                q_proj = x_1_pred - dot_pq * x_sphere
                q_proj_norm = torch.norm(q_proj, dim=-1, keepdim=True).clamp(min=1e-8)
                dot_clamped = dot_pq.clamp(-1 + 1e-7, 1 - 1e-7)
                dist = torch.acos(dot_clamped)
                tangent = q_proj / q_proj_norm * dist * step_weight

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

        # Log step losses and distances
        step_log = {f"diag/step_{i}_loss": step_losses[i] for i in range(steps)}
        dist_log = {f"diag/step_{i}_geodist": step_distances[i] for i in range(steps)}

        # Final eval (standard)
        final_probs = sphere_to_simplex(x_sphere)
        final_logits = torch.log(final_probs.clamp(min=1e-10))

        token_nll = F.cross_entropy(
            final_logits.transpose(1, 2), input_ids, reduction="none"
        )
        token_nll = token_nll * loss_mask.float()

        self.meter.update(
            split="eval",
            value=token_nll.detach(),
            weight=loss_mask.float().detach(),
        )

        loss = token_nll.sum() / loss_mask.sum().clamp_min(1)

        # Log all diagnostics
        all_logs = {**bucket_log, **step_log, **dist_log}
        all_logs["diag/final_eval_loss"] = loss.item()

        # Only log on main process
        if self.accelerator.is_main_process:
            self.log(all_logs)

            # Also print summary
            if self._diagnostic_step % 5 == 0:
                print(f"\n=== Diagnostic Summary (eval step {self._diagnostic_step}) ===")
                print(f"Loss by time bucket (training-style):")
                for i in range(10):
                    print(f"  t={i*10}-{(i+1)*10}%: {bucket_losses[i]:.4f}")
                print(f"\nLoss by integration step (eval-style):")
                for i in range(0, steps, 4):
                    print(f"  step {i}: loss={step_losses[i]:.4f}, geodist={step_distances[i]:.4f}")
                print(f"\nFinal eval loss: {loss.item():.4f}")
                print("=" * 50)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), final_logits.detach().contiguous(), labels.detach().contiguous())


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
