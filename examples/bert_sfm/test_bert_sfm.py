"""
Quick test script for BERT SFM training + sampling.

Usage:
    python scripts/test_bert_sfm.py
"""

import torch
import transformers
from datasets import Dataset
from tqdm import tqdm

from dllm.pipelines.bert_sfm import BertSFMTrainer, BertSFMSampler, BertSFMSamplerConfig

print("=== BERT SFM Integration Test ===\n")

# Load tiny model
model_name = "prajjwal1/bert-tiny"
print(f"Loading model: {model_name}")
model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# Overfit on a single pattern
train_texts = ["The cat sat on the mat."] * 20


def tokenize_fn(examples):
    tokens = tokenizer(
        examples["text"], truncation=True, max_length=32, padding=False  # No padding
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = Dataset.from_dict({"text": train_texts})
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
dataset.set_format("torch")

# Debug: show what the training data looks like
print(f"\n=== Training data sample ===")
sample = dataset[0]
print(f"Input IDs: {sample['input_ids'].tolist()}")
print(f"Decoded: {tokenizer.decode(sample['input_ids'])}")

# Train
args = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm",
    per_device_train_batch_size=20,
    num_train_epochs=10,
    learning_rate=5e-3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
)

trainer = BertSFMTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

print("\n=== Training ===")
result = trainer.train()
print(f"Final loss: {result.training_loss:.4f}")

# Test sampling
print("\n=== Testing Sampling ===")
model.eval()
sampler = BertSFMSampler(model=model, tokenizer=tokenizer)

test_text = "The [MASK] sat on the [MASK]."
inputs = tokenizer(test_text, return_tensors="pt")
print(f"Input: {test_text}")

for steps in [20, 50]:
    config = BertSFMSamplerConfig(steps=steps, temperature=0.0)
    output = sampler.infill([inputs["input_ids"][0]], config=config)
    output_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Steps={steps:3d}: {output_text}")

# Direct MLM prediction for comparison
print("\n=== Direct MLM Prediction ===")
with torch.no_grad():
    outputs = model(**inputs)
mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).squeeze()
preds = outputs.logits[0, mask_pos].argmax(dim=-1)
print(f"Direct prediction: {tokenizer.decode(preds)}")

# Check top-5 predictions at mask positions
print("\n=== Top-5 predictions at mask positions ===")
for i, pos in enumerate(mask_pos.nonzero().squeeze().tolist()):
    if not isinstance(pos, list):
        pos = [pos] if isinstance(pos, int) else pos.tolist()
    for p in (pos if isinstance(pos, list) else [pos]):
        logits = outputs.logits[0, p]
        top5 = logits.topk(5)
        tokens = [tokenizer.decode([idx]) for idx in top5.indices.tolist()]
        print(f"Position {p}: {tokens}")

# Test with soft embedding input (like training)
print("\n=== Test with soft embeddings (like training) ===")
import torch.nn.functional as F
from dllm.pipelines.bert_sfm.sampler import simplex_to_sphere, sphere_to_simplex

# Get embedding layer
embed_layer = model.get_input_embeddings()
vocab_size = model.config.vocab_size

# Convert input to one-hot -> sphere -> soft embeddings
input_ids = inputs["input_ids"]
one_hot = F.one_hot(input_ids, num_classes=vocab_size).float()
x_sphere = simplex_to_sphere(one_hot)
soft_emb = torch.matmul(x_sphere, embed_layer.weight)

with torch.no_grad():
    soft_outputs = model(inputs_embeds=soft_emb)

soft_preds = soft_outputs.logits[0].argmax(dim=-1)
print(f"Soft embedding prediction: {tokenizer.decode(soft_preds)}")

print("\n=== Test Complete ===")
