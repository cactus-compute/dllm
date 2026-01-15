#!/usr/bin/env python3
"""
Sampling script for BERT-RDLM.

Generate text samples from a trained BERT-RDLM model.

Usage:
    python examples/bert_rdlm/sample.py \
        --model_path models/bert_rdlm/mixture_geometric/checkpoint-final \
        --prompt "The meaning of life is" \
        --n_samples 5 \
        --n_steps 100
"""

import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from dllm.pipelines.bert_rdlm import (
    BertRDLMSampler,
    BertRDLMSamplerConfig,
    RDLMSchedule,
    RDLMScheduleConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from BERT-RDLM model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/bert_rdlm/mixture_geometric/checkpoint-final",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Below is an instruction. Write a response.\n\n### Instruction:\nWhat is machine learning?\n\n### Response:\n",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of integration steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mixture",
        choices=["uniform", "masked", "mixture"],
        help="Prior distribution type",
    )
    parser.add_argument(
        "--schedule_type",
        type=str,
        default="geometric",
        choices=["geometric", "linear", "cosine"],
        help="Noise schedule type",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable stochastic sampling (use deterministic integration)",
    )
    parser.add_argument(
        "--mix_type",
        type=str,
        default="step",
        choices=["linear", "sqrt", "step"],
        help="Mixture schedule type",
    )
    parser.add_argument(
        "--mix_step_thr",
        type=float,
        default=0.0,
        help="Step threshold for mix_type=step",
    )
    parser.add_argument(
        "--sampling_eps",
        type=float,
        default=1e-5,
        help="Sampling epsilon to avoid t=1",
    )
    parser.add_argument(
        "--no_mask_token",
        action="store_true",
        help="Disable extra mask token dimension (RDLM default is to add one)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for sampling",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Vocab size: {model.config.vocab_size}")

    # Create sampler
    sampler = BertRDLMSampler(model=model, tokenizer=tokenizer)

    # Create sampler config
    config = BertRDLMSamplerConfig(
        n_steps=args.n_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prior_type=args.prior_type,
        schedule_type=args.schedule_type,
        stochastic=not args.deterministic,
        mix_type=args.mix_type,
        mix_step_thr=args.mix_step_thr,
        sampling_eps=args.sampling_eps,
        add_mask_token=not args.no_mask_token,
    )

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors="pt")[0]
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # Generate samples
    print(f"\nGenerating {args.n_samples} samples with {args.n_steps} steps...")
    print("-" * 50)

    inputs = [prompt_tokens.clone() for _ in range(args.n_samples)]

    with torch.no_grad():
        output = sampler.sample(
            inputs=inputs,
            config=config,
            return_dict=False,
        )

    # Decode and print samples
    for i, seq in enumerate(output):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        # Extract just the response part after the prompt
        if "### Response:" in text:
            response_start = text.find("### Response:") + len("### Response:")
            response = text[response_start:].strip()
        else:
            response = text[len(args.prompt):].strip()

        print(f"\n=== Sample {i + 1} ===")
        print(response)
        print("-" * 50)


def infill_demo():
    """Demo of infilling (fill in [MASK] tokens)."""
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Create sampler
    sampler = BertRDLMSampler(model=model, tokenizer=tokenizer)

    config = BertRDLMSamplerConfig(
        n_steps=args.n_steps,
        temperature=args.temperature,
        prior_type=args.prior_type,
        schedule_type=args.schedule_type,
        stochastic=not args.deterministic,
        mix_type=args.mix_type,
        mix_step_thr=args.mix_step_thr,
        sampling_eps=args.sampling_eps,
        add_mask_token=not args.no_mask_token,
    )

    # Example with masks
    text_with_masks = "The capital of France is [MASK]. It is known for the [MASK] Tower."
    print(f"\nInput: {text_with_masks}")

    tokens = tokenizer.encode(text_with_masks, return_tensors="pt")[0]
    inputs = [tokens.clone() for _ in range(args.n_samples)]

    with torch.no_grad():
        output = sampler.infill(
            inputs=inputs,
            config=config,
            return_dict=False,
        )

    print("\nInfilled outputs:")
    for i, seq in enumerate(output):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"  {i + 1}: {text}")


if __name__ == "__main__":
    main()
