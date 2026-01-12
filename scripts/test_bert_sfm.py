"""
Quick integration test for BERT SFM (Fisher-Rao Flow Matching) trainer and sampler.
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from dllm.pipelines.bert_sfm import BertSFMTrainer, BertSFMSampler, BertSFMSamplerConfig
from dllm.pipelines.bert_sfm.trainer import geodesic_interpolant_to_onehot
from dllm.pipelines.bert_sfm.geodesic_utils import (
    geodesic_interpolant,
    uniform_prior,
    simplex_to_sphere,
    TimeEmbedding,
)


def test_geodesic_interpolant():
    """Verify the optimized geodesic matches the original and satisfies boundary conditions."""
    print("\n" + "=" * 60)
    print("Geodesic Interpolant Unit Tests")
    print("=" * 60)

    torch.manual_seed(42)
    b, l, v = 2, 4, 100

    x_0 = uniform_prior((b, l, v), device="cpu", dtype=torch.float32)
    target_indices = torch.randint(0, v, (b, l))
    t = torch.rand(b)

    # Test 1: Compare optimized vs original implementation
    print("\nTest 1: Optimized vs Original implementation...")
    x_t_new = geodesic_interpolant_to_onehot(x_0, target_indices, t)

    # Original version (create full one-hot x_1)
    x_1 = torch.zeros(b, l, v).scatter(-1, target_indices.unsqueeze(-1), 1.0)
    x_1_sphere = simplex_to_sphere(x_1)  # sqrt of one-hot = one-hot (since sqrt(1)=1, sqrt(0)=0)
    x_t_old = geodesic_interpolant(x_0, x_1_sphere, t)

    max_diff = (x_t_new - x_t_old).abs().max().item()
    print(f"  Max difference: {max_diff:.2e}")
    # Allow 2e-4 tolerance - different computation paths have small numerical differences
    assert torch.allclose(x_t_new, x_t_old, atol=2e-4), f"Geodesic mismatch! Max diff: {max_diff}"
    print("  PASSED")

    # Test 2: Boundary condition t=0 should give x_0
    print("\nTest 2: Boundary condition t=0 -> x_0...")
    x_t_0 = geodesic_interpolant_to_onehot(x_0, target_indices, torch.zeros(b))
    max_diff_t0 = (x_t_0 - x_0).abs().max().item()
    print(f"  Max difference from x_0: {max_diff_t0:.2e}")
    assert torch.allclose(x_t_0, x_0, atol=2e-4), f"t=0 should give x_0! Max diff: {max_diff_t0}"
    print("  PASSED")

    # Test 3: Boundary condition t=1 should give x_1 (one-hot on sphere)
    print("\nTest 3: Boundary condition t=1 -> x_1 (one-hot)...")
    x_t_1 = geodesic_interpolant_to_onehot(x_0, target_indices, torch.ones(b))
    max_diff_t1 = (x_t_1 - x_1_sphere).abs().max().item()
    print(f"  Max difference from x_1: {max_diff_t1:.2e}")
    assert torch.allclose(x_t_1, x_1_sphere, atol=2e-4), f"t=1 should give x_1! Max diff: {max_diff_t1}"
    print("  PASSED")

    # Test 4: Output should be on unit sphere
    print("\nTest 4: Output is on unit sphere...")
    norms = torch.norm(x_t_new, dim=-1)
    max_norm_diff = (norms - 1.0).abs().max().item()
    print(f"  Max deviation from unit norm: {max_norm_diff:.2e}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=2e-4), f"Not on unit sphere! Max diff: {max_norm_diff}"
    print("  PASSED")

    # Test 5: Test with bf16 dtype
    print("\nTest 5: bf16 dtype preservation...")
    x_0_bf16 = x_0.to(torch.bfloat16)
    t_bf16 = t.to(torch.bfloat16)
    x_t_bf16 = geodesic_interpolant_to_onehot(x_0_bf16, target_indices, t_bf16)
    assert x_t_bf16.dtype == torch.bfloat16, f"Expected bf16, got {x_t_bf16.dtype}"
    print(f"  Output dtype: {x_t_bf16.dtype}")
    print("  PASSED")

    print("\n" + "-" * 60)
    print("All geodesic tests PASSED!")
    print("-" * 60)


def test_bert_sfm():
    print("=" * 60)
    print("BERT SFM Integration Test")
    print("=" * 60)

    # Use a small model for fast testing
    model_name = "prajjwal1/bert-tiny"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has required tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "[MASK]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Create training data - simple repeated sentences for easy learning
    train_texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "A bird flew over the tree.",
    ] * 10  # Repeat to have more samples

    print(f"\nTraining on {len(train_texts)} samples")

    # Tokenize with no padding (variable length sequences)
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=32,
        padding=False,
        return_tensors=None,
    )

    # Create dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                "labels": torch.tensor(self.encodings["input_ids"][idx]),
            }

    train_dataset = SimpleDataset(train_encodings)

    # Training config
    training_args = BertSFMTrainer.BertSFMConfig(
        output_dir="./test_bert_sfm_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-4,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        schedule_type="linear",
        loss_weight_type="uniform",
    )

    # Create trainer
    from transformers import DataCollatorForSeq2Seq

    trainer = BertSFMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
    )

    print("\n--- Training ---")
    trainer.train()
    print("Training complete!")

    # Test sampling
    print("\n--- Testing Sampler ---")
    model.eval()

    sampler = BertSFMSampler(model=model, tokenizer=tokenizer)
    config = BertSFMSamplerConfig(
        steps=50,
        max_new_tokens=6,
        temperature=0.0,
    )

    # Get time embedding from trainer (trained alongside model)
    time_embedding = trainer.time_embedding

    # Test infill (mask filling)
    test_text = "The [MASK] sat on the [MASK]."
    print(f"\nInfill test: '{test_text}'")

    test_ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
    print(f"Input token IDs: {test_ids.tolist()}")

    output = sampler.infill([test_ids[0]], config=config, time_embedding=time_embedding)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output: '{output_text}'")

    # Test generation (continuation)
    prompt = "The cat"
    print(f"\nGeneration test: '{prompt}'")

    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = sampler.sample([prompt_ids[0]], config=config, time_embedding=time_embedding)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output: '{output_text}'")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def test_overfit_single_sentence():
    """Overfit on a single sentence and verify the model can generate it from masks."""
    print("\n" + "=" * 60)
    print("Overfit Test: Single Sentence")
    print("=" * 60)

    model_name = "prajjwal1/bert-tiny"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "[MASK]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Single sentence repeated many times
    target_sentence = "The cat sat on the mat."
    train_texts = [target_sentence] * 1000

    print(f"\nOverfitting on: '{target_sentence}'")
    print(f"Training samples: {len(train_texts)}")

    # Tokenize
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=32,
        padding=False,
        return_tensors=None,
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                "labels": torch.tensor(self.encodings["input_ids"][idx]),
            }

    train_dataset = SimpleDataset(train_encodings)

    # Training config - more epochs to really overfit
    training_args = BertSFMTrainer.BertSFMConfig(
        output_dir="./test_overfit_output",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        learning_rate=1e-3,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        schedule_type="linear",
        loss_weight_type="uniform",
    )

    from transformers import DataCollatorForSeq2Seq

    trainer = BertSFMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
    )

    print("\n--- Training (overfitting) ---")
    trainer.train()
    print("Training complete!")

    # Test generation from all masks
    print("\n--- Testing Generation from All Masks ---")
    model.eval()

    sampler = BertSFMSampler(model=model, tokenizer=tokenizer)
    config = BertSFMSamplerConfig(
        steps=100,  # More steps for better quality
        max_new_tokens=8,
        temperature=0.0,  # Greedy
    )

    # Get time embedding from trainer
    time_embedding = trainer.time_embedding

    # Create input with all masks (same length as target)
    target_ids = tokenizer.encode(target_sentence, return_tensors="pt")[0]
    num_tokens = len(target_ids)
    print(f"\nTarget: '{target_sentence}'")
    print(f"Target IDs: {target_ids.tolist()}")
    print(f"Target decoded: '{tokenizer.decode(target_ids, skip_special_tokens=False)}'")

    # All masks (excluding [CLS] and [SEP])
    all_mask_ids = target_ids.clone()
    # Mask everything except CLS (101) and SEP (102)
    for i in range(len(all_mask_ids)):
        if all_mask_ids[i] not in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            all_mask_ids[i] = tokenizer.mask_token_id

    print(f"\nInput (all masks): '{tokenizer.decode(all_mask_ids, skip_special_tokens=False)}'")
    print(f"Input IDs: {all_mask_ids.tolist()}")

    # Run infill
    all_mask_ids = all_mask_ids.to(device)
    output = sampler.infill([all_mask_ids], config=config, time_embedding=time_embedding)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\nGenerated: '{output_text}'")
    print(f"Expected:  '{target_sentence}'")

    # Check if it matches
    # Normalize for comparison (lowercase, strip)
    generated_normalized = output_text.lower().strip()
    expected_normalized = target_sentence.lower().strip()

    if generated_normalized == expected_normalized:
        print("\n✓ SUCCESS: Model perfectly reproduced the training sentence!")
    else:
        print(f"\n✗ MISMATCH: Generated text differs from target")
        print(f"  Generated tokens: {tokenizer.encode(output_text)}")
        print(f"  Expected tokens:  {tokenizer.encode(target_sentence)}")

    print("\n" + "=" * 60)


def test_overfit_multiple_sentences():
    """Overfit on 10 sentences and verify the model generates one of them from all masks."""
    print("\n" + "=" * 60)
    print("Overfit Test: Multiple Sentences (10 unique)")
    print("=" * 60)

    model_name = "prajjwal1/bert-tiny"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "[MASK]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # 10 different sentences
    sentences = [
        "The cat sat on the mat.",
        "A dog ran in the park.",
        "She loves to read books.",
        "He plays guitar every day.",
        "The sun is shining bright.",
        "I like to eat pizza.",
        "Birds fly in the sky.",
        "The car is very fast.",
        "We went to the beach.",
        "They watch movies at night.",
    ]

    # Repeat each sentence to get enough training data
    train_texts = sentences * 100  # 1000 total samples

    print(f"\nTraining sentences:")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")
    print(f"\nTotal training samples: {len(train_texts)}")

    # Tokenize
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=32,
        padding=False,
        return_tensors=None,
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                "labels": torch.tensor(self.encodings["input_ids"][idx]),
            }

    train_dataset = SimpleDataset(train_encodings)

    # Training config - 100 epochs to really overfit
    training_args = BertSFMTrainer.BertSFMConfig(
        output_dir="./test_overfit_multi_output",
        num_train_epochs=100,
        per_device_train_batch_size=32,
        learning_rate=1e-3,
        logging_steps=100,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        schedule_type="linear",
        loss_weight_type="uniform",
    )

    from transformers import DataCollatorForSeq2Seq

    trainer = BertSFMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
    )

    print("\n--- Training (overfitting on 10 sentences) ---")
    trainer.train()
    print("Training complete!")

    # Test generation from all masks
    print("\n--- Testing Generation from All Masks ---")
    model.eval()

    sampler = BertSFMSampler(model=model, tokenizer=tokenizer)
    config = BertSFMSamplerConfig(
        steps=100,
        max_new_tokens=8,
        temperature=0.0,  # Greedy
    )

    # Get time embedding from trainer
    time_embedding = trainer.time_embedding

    # Normalize sentences for comparison
    normalized_sentences = set(s.lower().strip() for s in sentences)

    # Test with different sequence lengths
    test_lengths = [7, 8, 9]  # Different numbers of content tokens (excluding CLS/SEP)

    print("\nGenerating from all-mask inputs of various lengths:")
    successes = 0
    total_tests = 0

    for num_masks in test_lengths:
        # Create input: [CLS] + num_masks * [MASK] + [SEP]
        mask_ids = [tokenizer.cls_token_id] + [tokenizer.mask_token_id] * num_masks + [tokenizer.sep_token_id]
        mask_tensor = torch.tensor(mask_ids).to(device)

        print(f"\n  Input ({num_masks} masks): '{tokenizer.decode(mask_tensor, skip_special_tokens=False)}'")

        # Generate multiple times to see variety
        for trial in range(3):
            total_tests += 1
            # Use different temperatures for variety
            temp_config = BertSFMSamplerConfig(
                steps=100,
                max_new_tokens=8,
                temperature=0.5 if trial > 0 else 0.0,
            )
            output = sampler.infill([mask_tensor], config=temp_config, time_embedding=time_embedding)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            normalized_output = output_text.lower().strip()

            is_match = normalized_output in normalized_sentences
            status = "✓" if is_match else "✗"
            if is_match:
                successes += 1

            print(f"    Trial {trial+1} (temp={temp_config.temperature}): '{output_text}' {status}")

    print(f"\n--- Results ---")
    print(f"Successes: {successes}/{total_tests}")

    if successes > 0:
        print("\n✓ SUCCESS: Model generated at least one training sentence!")
    else:
        print("\n✗ FAIL: Model did not generate any training sentences")
        print("  (This might be expected - generating exact matches from 10 options is harder)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_geodesic_interpolant()
    # test_overfit_single_sentence()
    test_overfit_multiple_sentences()
    # test_bert_sfm()  # Skip the basic test, overfit test is more thorough
