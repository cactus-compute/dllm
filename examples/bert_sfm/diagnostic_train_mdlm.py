#!/usr/bin/env python3
"""
Diagnostic training script for MDLM baseline.

This script mirrors diagnostic_train.py for Spherical Flow Matching (SFM) but uses
the MDLM (Masked Diffusion Language Model) objective instead. This allows direct
comparison between the two approaches on the same dataset.

Metrics Comparison Guide:
========================

DIRECTLY COMPARABLE (same semantics):
- eval_final_loss: Generation loss - the key metric!
  * SFM: -log(integrated_probs[ground_truth]) averaged over tokens
  * MDLM: -log(P(ground_truth | context)) accumulated over unmasking steps
  * Both answer: "How much probability does the model assign to the correct sequence?"
  * Lower is better, comparable magnitudes, fair comparison

- eval_loss_t_{bucket}: Loss at different corruption levels
  * SFM: Loss at time t (soft interpolation between noise and target)
  * MDLM: Loss at mask rate % (discrete masking)
  * Both measure: "How hard is reconstruction at this corruption level?"

- eval_step_{i}_entropy / eval_step_{i}_entropy_pct: Prediction uncertainty
  * Both measure model confidence at each generation step
  * Low = confident, high = uncertain

COMPARABLE WITH CAVEATS:
- eval_step_{i}_loss: Per-step CE loss
  * SFM: CE from soft sphere embeddings → endpoint prediction
  * MDLM: CE from masked tokens → reconstruction
  * Direction same (lower=better), but different input representations

- train_loss: Training loss
  * SFM: CE loss predicting endpoint from interpolated x_t
  * MDLM: CE loss predicting original from masked input
  * Different parameterizations; absolute values not directly comparable
  * Trend should be similar (both should decrease)

NOT COMPARABLE (SFM-specific geometry):
- eval_step_{i}_geodist: Geodesic distance on sphere
  * MDLM has no continuous manifold trajectory

- eval_step_{i}_tv: Total variation between consecutive soft distributions
  * MDLM uses discrete tokens, not soft distributions

- eval_step_{i}_contraction: Local contraction ratio
  * Manifold-specific concept; no MDLM equivalent

- mse_loss, geodesic_loss: SFM auxiliary losses

MDLM-SPECIFIC:
- eval_step_{i}_mask_rate: % of tokens still masked at step i
  * No SFM equivalent (SFM doesn't have discrete masking)

Usage:
    python examples/bert_sfm/diagnostic_train_mdlm.py
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
from dllm.core.trainers.mdlm import MDLMTrainer
from dllm.core.schedulers import LinearAlphaScheduler

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
class TrainingArguments(MDLMTrainer.MDLMConfig):
    output_dir: str = "models/diagnostic/mdlm_baseline"
    group_by_length: bool = True
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 16
    # More frequent eval for diagnostics
    eval_steps: float = 0.05  # Every 5%
    # MDLM-specific
    loss_weight_type: str = "uniform"  # Match SFM default
    loss_norm_type: str = "token"
    # Number of unmasking steps for evaluation (like SFM's integration steps)
    eval_unmasking_steps: int = 20


class DiagnosticMDLMTrainer(MDLMTrainer):
    """Extended MDLM trainer with diagnostic logging during evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_unmasking_steps = getattr(self.args, 'eval_unmasking_steps', 20)

    def _init_diagnostic_accumulators(self):
        """Initialize accumulators for averaging metrics across eval batches."""
        steps = self.eval_unmasking_steps
        self._diag_bucket_losses = [[] for _ in range(10)]  # 10 masking rate buckets
        self._diag_step_losses = [[] for _ in range(steps)]
        self._diag_step_entropies = [[] for _ in range(steps)]  # Entropy per step
        self._diag_step_mask_rates = [[] for _ in range(steps)]  # Remaining mask % per step
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
            steps = self.eval_unmasking_steps

            # Average bucket losses
            avg_bucket_losses = [
                sum(self._diag_bucket_losses[i]) / len(self._diag_bucket_losses[i])
                if self._diag_bucket_losses[i] else 0.0
                for i in range(10)
            ]

            # Average step losses and entropies
            avg_step_losses = [
                sum(self._diag_step_losses[i]) / len(self._diag_step_losses[i])
                if self._diag_step_losses[i] else 0.0
                for i in range(steps)
            ]
            avg_step_entropies = [
                sum(self._diag_step_entropies[i]) / len(self._diag_step_entropies[i])
                if self._diag_step_entropies[i] else 0.0
                for i in range(steps)
            ]
            avg_step_mask_rates = [
                sum(self._diag_step_mask_rates[i]) / len(self._diag_step_mask_rates[i])
                if self._diag_step_mask_rates[i] else 0.0
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
            all_logs = {}
            for i in range(10):
                all_logs[f"eval_loss_t_{i*10}-{(i+1)*10}pct"] = avg_bucket_losses[i]
            for i in range(steps):
                all_logs[f"eval_step_{i}_loss"] = avg_step_losses[i]
                all_logs[f"eval_step_{i}_entropy"] = avg_step_entropies[i]
                all_logs[f"eval_step_{i}_entropy_pct"] = 100.0 * avg_step_entropies[i] / max_entropy
                all_logs[f"eval_step_{i}_mask_rate"] = avg_step_mask_rates[i]
            all_logs["eval_final_loss"] = avg_final_loss
            all_logs["eval_mean_step_entropy"] = sum(avg_step_entropies) / len(avg_step_entropies) if avg_step_entropies else 0.0
            all_logs["eval_mean_step_entropy_pct"] = 100.0 * all_logs["eval_mean_step_entropy"] / max_entropy

            # Log once per evaluation
            self.log(all_logs)

            # Print summary
            print(f"\n=== MDLM Diagnostic Summary (averaged over {self._diag_batch_count} batches) ===")
            print(f"Training objective: Masked Diffusion Language Modeling (MDLM)")
            print(f"Unmasking steps: {steps}")
            print(f"\nLoss by masking rate bucket (training-style):")
            for i in range(10):
                print(f"  mask={i*10}-{(i+1)*10}%: {avg_bucket_losses[i]:.4f}")
            print(f"\nLoss by unmasking step (eval-style):")
            for i in range(0, steps, 4):
                entropy_pct = 100.0 * avg_step_entropies[i] / max_entropy
                mask_rate = avg_step_mask_rates[i]
                print(f"  step {i}: loss={avg_step_losses[i]:.4f}, entropy={entropy_pct:.1f}%, mask_rate={mask_rate:.1f}%")
            print(f"\nFinal eval loss: {avg_final_loss:.4f}")
            mean_entropy_pct = 100.0 * sum(avg_step_entropies) / len(avg_step_entropies) / max_entropy if avg_step_entropies else 0.0
            print(f"Mean step entropy: {mean_entropy_pct:.1f}% of max (low = confident, high = uncertain)")
            print("=" * 50)

        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Extended evaluation with diagnostic logging.

        Unlike SFM which integrates a continuous flow, MDLM progressively unmasks tokens.
        We simulate this by:
        1. Starting with all response tokens masked
        2. At each step, unmask a fraction of tokens based on model confidence
        3. Measure CE loss at each step
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        unwrapped_model = model.module if hasattr(model, "module") else model
        vocab_size = unwrapped_model.config.vocab_size
        device = input_ids.device

        loss_mask = labels != -100  # Positions to generate (not prompt)
        num_to_generate = loss_mask.sum(dim=1)  # Tokens per sequence to generate

        # ============================================================
        # DIAGNOSTIC 1: Loss bucketed by masking rate (like SFM's time buckets)
        # ============================================================
        bucket_losses = []

        for i in range(10):
            # Masking rate in this bucket
            mask_rate_low = i * 0.1
            mask_rate_high = (i + 1) * 0.1
            mask_rate = mask_rate_low + (mask_rate_high - mask_rate_low) * torch.rand(b, device=device)

            # Create masked input
            # Only mask positions where loss_mask=True
            mask_probs = torch.rand(b, l, device=device)
            should_mask = (mask_probs < mask_rate.unsqueeze(1)) & loss_mask

            masked_input_ids = torch.where(
                should_mask,
                self.processing_class.mask_token_id,
                input_ids
            )

            with torch.no_grad():
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                token_loss = F.cross_entropy(
                    logits.transpose(1, 2), input_ids, reduction="none"
                )
                # Loss only on masked positions
                masked_loss = (token_loss * should_mask.float()).sum() / should_mask.sum().clamp_min(1)
                bucket_losses.append(masked_loss.item())

        # Accumulate bucket losses
        for i in range(10):
            self._diag_bucket_losses[i].append(bucket_losses[i])

        # ============================================================
        # DIAGNOSTIC 2: Progressive unmasking (eval-style)
        # ============================================================
        steps = self.eval_unmasking_steps
        step_losses = []
        step_entropies = []
        step_mask_rates = []

        # Start with all response tokens masked
        current_ids = input_ids.clone()
        current_ids[loss_mask] = self.processing_class.mask_token_id

        # Track which positions are still masked
        still_masked = loss_mask.clone()

        for step_idx in range(steps):
            # How many tokens to unmask this step
            # Linear schedule: unmask equal fraction each step
            total_masked = still_masked.sum(dim=1).float()
            tokens_per_step = num_to_generate.float() / steps

            with torch.no_grad():
                # Forward pass
                outputs = model(input_ids=current_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)

                # Compute loss on still-masked positions
                step_ce = F.cross_entropy(
                    logits.transpose(1, 2), input_ids, reduction="none"
                )
                masked_step_loss = (step_ce * still_masked.float()).sum() / still_masked.sum().clamp_min(1)
                step_losses.append(masked_step_loss.item())

                # Compute entropy on still-masked positions
                entropy_per_pos = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
                mean_entropy = (entropy_per_pos * still_masked.float()).sum() / still_masked.sum().clamp_min(1)
                step_entropies.append(mean_entropy.item())

                # Track mask rate
                mask_rate = 100.0 * still_masked.sum().float() / loss_mask.sum().clamp_min(1)
                step_mask_rates.append(mask_rate.item())

                # Unmask tokens with highest confidence (lowest entropy)
                # Get predictions for masked positions
                predicted_tokens = logits.argmax(dim=-1)

                # Confidence score: max probability
                confidence = probs.max(dim=-1).values  # [b, l]

                # For each sequence, unmask the most confident masked positions
                for batch_idx in range(b):
                    batch_masked = still_masked[batch_idx]
                    if not batch_masked.any():
                        continue

                    # Get confidence at masked positions
                    masked_positions = batch_masked.nonzero(as_tuple=True)[0]
                    masked_confidence = confidence[batch_idx, masked_positions]

                    # Number to unmask this step (at least 1)
                    n_unmask = max(1, int(tokens_per_step[batch_idx].item()))
                    n_unmask = min(n_unmask, len(masked_positions))

                    # Get indices of most confident
                    _, top_indices = masked_confidence.topk(n_unmask)
                    positions_to_unmask = masked_positions[top_indices]

                    # Unmask: replace mask token with predicted token
                    current_ids[batch_idx, positions_to_unmask] = predicted_tokens[batch_idx, positions_to_unmask]
                    still_masked[batch_idx, positions_to_unmask] = False

        # Accumulate step metrics
        for i in range(steps):
            self._diag_step_losses[i].append(step_losses[i])
            self._diag_step_entropies[i].append(step_entropies[i])
            self._diag_step_mask_rates[i].append(step_mask_rates[i])

        # ============================================================
        # Final evaluation: compute generation loss
        # ============================================================
        # IMPORTANT: For fair comparison with SFM, we need to measure how much
        # probability mass the model assigns to the GROUND TRUTH tokens.
        #
        # SFM computes: -log(final_probs[ground_truth_token])
        # where final_probs comes from the integrated soft distribution.
        #
        # For MDLM, we track the cumulative log-probability of choosing the
        # correct token at each unmasking step. This is equivalent to:
        # P(correct sequence) = prod over steps of P(correct token | context)
        #
        # But there's a subtlety: MDLM unmasks greedily (most confident first),
        # so we can't directly compute P(ground_truth). Instead, we use a
        # different but comparable metric:
        #
        # "Oracle unmasking loss": At each step, given the tokens unmasked so far
        # (using GROUND TRUTH, not predictions), what's the CE loss on remaining
        # masked positions? This measures how well the model would generate if
        # it had made all correct choices so far.
        #
        # This is the fairest comparison because:
        # - SFM: soft distribution quality → probability of correct token
        # - MDLM: discrete generation quality → probability of correct token
        #
        # We compute this by re-running unmasking with oracle (ground truth) tokens.

        with torch.no_grad():
            # Re-run with oracle unmasking to get fair generation loss
            oracle_ids = input_ids.clone()
            oracle_ids[loss_mask] = self.processing_class.mask_token_id
            oracle_still_masked = loss_mask.clone()

            cumulative_nll = torch.zeros(b, device=device)
            tokens_scored = torch.zeros(b, device=device)

            for step_idx in range(steps):
                if not oracle_still_masked.any():
                    break

                outputs = model(input_ids=oracle_ids, attention_mask=attention_mask)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)

                # Get log prob of ground truth at each masked position
                gt_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

                # Confidence for unmasking order (use model's confidence, not oracle)
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1).values

                for batch_idx in range(b):
                    batch_masked = oracle_still_masked[batch_idx]
                    if not batch_masked.any():
                        continue

                    masked_positions = batch_masked.nonzero(as_tuple=True)[0]
                    masked_confidence = confidence[batch_idx, masked_positions]

                    n_unmask = max(1, int((num_to_generate[batch_idx].float() / steps).item()))
                    n_unmask = min(n_unmask, len(masked_positions))

                    _, top_indices = masked_confidence.topk(n_unmask)
                    positions_to_unmask = masked_positions[top_indices]

                    # Accumulate NLL for these positions (this is the generation cost)
                    cumulative_nll[batch_idx] -= gt_log_probs[batch_idx, positions_to_unmask].sum()
                    tokens_scored[batch_idx] += n_unmask

                    # Unmask with GROUND TRUTH (oracle)
                    oracle_ids[batch_idx, positions_to_unmask] = input_ids[batch_idx, positions_to_unmask]
                    oracle_still_masked[batch_idx, positions_to_unmask] = False

            # Average NLL per token (comparable to SFM's eval_final_loss)
            final_loss = cumulative_nll.sum() / tokens_scored.sum().clamp_min(1)

        self._diag_final_losses.append(final_loss.item())
        self._diag_batch_count += 1

        # Update metrics for standard trainer tracking
        # Use the oracle-based NLL for fair PPL computation
        per_token_nll = cumulative_nll / tokens_scored.clamp_min(1)
        expanded_nll = per_token_nll.unsqueeze(1).expand(b, l) * loss_mask.float()
        self.meter.update(
            split="eval",
            value=expanded_nll.detach(),
            weight=loss_mask.float().detach(),
        )

        if prediction_loss_only:
            return (final_loss.detach(), None, None)

        return (final_loss.detach(), final_logits.detach().contiguous(), labels.detach().contiguous())


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
    logger.info("Start diagnostic MDLM training...")

    trainer = DiagnosticMDLMTrainer(
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
