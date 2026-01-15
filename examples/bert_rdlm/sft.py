#!/usr/bin/env python3
"""
Production training script for BERT-RDLM (Riemannian Diffusion Language Model).

Launch with accelerate for multi-GPU or mixed precision training:

    # Single GPU with bf16
    accelerate launch --mixed_precision bf16 examples/bert_rdlm/sft.py

    # Multi-GPU
    accelerate launch --multi_gpu --num_processes 4 examples/bert_rdlm/sft.py

    # With custom config
    accelerate launch --config_file accelerate_config.yaml examples/bert_rdlm/sft.py

Example overnight training command:
    nohup accelerate launch --mixed_precision bf16 examples/bert_rdlm/sft.py \
        --output_dir models/bert_rdlm/overnight_run \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --dataset_args tatsu-lab/alpaca \
        > training.log 2>&1 &
"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import accelerate
import torch
import transformers

import dllm
from dllm.pipelines.bert_rdlm import BertRDLMTrainer
from dllm.pipelines.bert_rdlm.trainer import BertRDLMTrainerConfig

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    # Good overnight datasets:
    # - tatsu-lab/alpaca (52k examples, ~2-4 hours)
    # - Open-Orca/OpenOrca (4M examples, multi-day)
    # - teknium/OpenHermes-2.5 (1M examples, overnight)
    # - HuggingFaceH4/ultrachat_200k (200k, overnight)
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    load_preprocessed_data: bool = False
    skip_post_process: bool = False
    mask_prompt_loss: bool = True


@dataclass
class TrainingArguments(BertRDLMTrainerConfig):
    # Output
    output_dir: str = "models/bert_rdlm/overnight"
    run_name: Optional[str] = None  # For wandb

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs

    # Batch size - adjust based on GPU memory
    # For 24GB GPU: batch_size=16, grad_accum=4 -> effective batch=64
    # For 80GB GPU: batch_size=32, grad_accum=4 -> effective batch=128
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Learning rate schedule
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"

    # RDLM-specific configuration
    prior_type: str = "mixture"  # "uniform", "masked", or "mixture"
    mixing_prob: float = 0.5
    schedule_type: str = "geometric"
    sigma_0: float = 0.001
    sigma_T: float = 1.0
    n_time_steps: int = 1000
    use_riemannian_normal: bool = True

    # Loss configuration
    loss_type: str = "ce"  # "ce" or "mse"
    loss_norm_type: str = "token"
    embed_type: str = "spherical"

    # Time embedding (recommended for RDLM)
    use_time_embedding: bool = True
    time_embedding_scale: float = 30.0

    # Evaluation - 100 steps for realistic generation quality
    eval_strategy: str = "steps"
    eval_steps: float = 0.1  # Every 10% of training
    eval_integration_steps: int = 100  # Must match inference for true performance

    # Saving
    save_strategy: str = "steps"
    save_steps: float = 0.25  # Every 25% of training
    save_total_limit: int = 3  # Keep last 3 checkpoints
    save_only_model: bool = True

    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 10
    report_to: str = "wandb"  # or "tensorboard" or "none"

    # Mixed precision - bf16=True for DeepSpeed compatibility
    bf16: bool = True
    fp16: bool = False

    # Optimization
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Efficiency
    group_by_length: bool = True
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2

    # Reproducibility
    seed: int = 42

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set run name for wandb if not provided
    if training_args.run_name is None:
        training_args.run_name = f"rdlm_{training_args.prior_type}_{training_args.schedule_type}"

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # Load model and tokenizer
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Load and preprocess dataset
    logger.info(f"Loading dataset: {data_args.dataset_args}")
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

    logger.info(f"Train examples: {len(dataset['train']):,}")
    if "test" in dataset:
        logger.info(f"Eval examples: {len(dataset['test']):,}")

    # Handle group_by_length with Arrow columns
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

    # Log training configuration
    logger.info("=" * 50)
    logger.info("RDLM Training Configuration:")
    logger.info(f"  Prior type: {training_args.prior_type}")
    logger.info(f"  Schedule type: {training_args.schedule_type}")
    logger.info(f"  Use Riemannian normal: {training_args.use_riemannian_normal}")
    logger.info(f"  Loss type: {training_args.loss_type}")
    logger.info(f"  Time embedding: {training_args.use_time_embedding}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info("=" * 50)

    # Create trainer
    trainer = BertRDLMTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
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

    # Train
    logger.info("Starting training...")
    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from: {training_args.resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save final model
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    logger.info(f"Saving final model to: {final_dir}")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    train()
