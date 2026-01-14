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
from dllm.pipelines.bert_sfm.geodesic_utils import TimeEmbedding

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

# Train with CE loss (default)
print("\n=== Training with CE Loss (endpoint prediction) ===")
args_ce = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm_ce",
    per_device_train_batch_size=20,
    num_train_epochs=10,
    learning_rate=5e-3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    loss_type="ce",
)

trainer_ce = BertSFMTrainer(
    model=model,
    args=args_ce,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

result_ce = trainer_ce.train()
print(f"Final CE loss: {result_ce.training_loss:.4f}")

# Test evaluation (prediction_step)
print("\n=== Testing Evaluation (prediction_step) ===")
eval_dataset = dataset.select(range(5))  # Use 5 samples for eval
trainer_ce.eval_dataset = eval_dataset
eval_metrics = trainer_ce.evaluate()
print(f"Eval metrics: {eval_metrics}")
print("prediction_step test PASSED!")

# Train with MSE loss (velocity prediction)
print("\n=== Training with MSE Loss (velocity prediction) ===")
model_mse = transformers.AutoModelForMaskedLM.from_pretrained(model_name)

args_mse = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm_mse",
    per_device_train_batch_size=20,
    num_train_epochs=10,
    learning_rate=1e-4,  # Lower LR for MSE (larger gradients)
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    loss_type="mse",
)

trainer_mse = BertSFMTrainer(
    model=model_mse,
    args=args_mse,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

result_mse = trainer_mse.train()
print(f"Final MSE loss: {result_mse.training_loss:.4f}")

# Test sampling with CE-trained model (endpoint prediction)
print("\n=== Testing Sampling (CE model, endpoint prediction) ===")
model.eval()
sampler_ce = BertSFMSampler(model=model, tokenizer=tokenizer)

# Get the time embedding from the trainer (trained alongside the model)
time_embedding_ce = trainer_ce.time_embedding

test_text = "The [MASK] sat on the [MASK]."
inputs = tokenizer(test_text, return_tensors="pt")
print(f"Input: {test_text}")

for steps in [20, 50]:
    config = BertSFMSamplerConfig(steps=steps, temperature=0.0, prediction_type="endpoint")
    output = sampler_ce.infill([inputs["input_ids"][0]], config=config, time_embedding=time_embedding_ce)
    output_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Steps={steps:3d}: {output_text}")

# Test sampling with MSE-trained model (velocity prediction)
print("\n=== Testing Sampling (MSE model, velocity prediction) ===")
model_mse.eval()
sampler_mse = BertSFMSampler(model=model_mse, tokenizer=tokenizer)

# Get the time embedding from the MSE trainer
time_embedding_mse = trainer_mse.time_embedding

print(f"Input: {test_text}")

for steps in [20, 50]:
    config = BertSFMSamplerConfig(steps=steps, temperature=0.0, prediction_type="velocity")
    output = sampler_mse.infill([inputs["input_ids"][0]], config=config, time_embedding=time_embedding_mse)
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
from dllm.pipelines.bert_sfm.geodesic_utils import simplex_to_sphere, sphere_to_simplex

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

# Test self-consistency training (noise mode - recommended)
print("\n=== Testing Self-Consistency Training (Noise Mode) ===")
model_sc = transformers.AutoModelForMaskedLM.from_pretrained(model_name)

args_sc = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm_self_consistency",
    per_device_train_batch_size=20,
    num_train_epochs=5,
    learning_rate=5e-3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    loss_type="ce",
    # Self-consistency settings (noise mode)
    self_consistency_prob=0.5,  # 50% of batches use self-consistency
    self_consistency_mode="noise",  # Tangent space noise injection
    self_consistency_noise_scale=0.1,
    self_consistency_schedule="constant",
)

trainer_sc = BertSFMTrainer(
    model=model_sc,
    args=args_sc,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

result_sc = trainer_sc.train()
print(f"Final self-consistency loss: {result_sc.training_loss:.4f}")
print("Self-consistency training test PASSED!")

# Test linear_ramp schedule with noise mode
print("\n=== Testing Self-Consistency with Linear Ramp Schedule ===")
model_sc_ramp = transformers.AutoModelForMaskedLM.from_pretrained(model_name)

args_sc_ramp = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm_self_consistency_ramp",
    per_device_train_batch_size=20,
    num_train_epochs=5,
    learning_rate=5e-3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    loss_type="ce",
    # Self-consistency with linear ramp (noise mode)
    self_consistency_prob=0.5,
    self_consistency_mode="noise",
    self_consistency_noise_scale=0.1,
    self_consistency_schedule="linear_ramp",
)

trainer_sc_ramp = BertSFMTrainer(
    model=model_sc_ramp,
    args=args_sc_ramp,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

result_sc_ramp = trainer_sc_ramp.train()
print(f"Final linear_ramp self-consistency loss: {result_sc_ramp.training_loss:.4f}")
print("Linear ramp self-consistency test PASSED!")

# Test training WITHOUT time embeddings (backward compatibility)
print("\n=== Testing Training WITHOUT Time Embeddings ===")
model_no_time = transformers.AutoModelForMaskedLM.from_pretrained(model_name)

args_no_time = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm_no_time",
    per_device_train_batch_size=20,
    num_train_epochs=5,
    learning_rate=5e-3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="no",
    use_cpu=True,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    loss_type="ce",
    use_time_embedding=False,  # Disable time embeddings
)

trainer_no_time = BertSFMTrainer(
    model=model_no_time,
    args=args_no_time,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

result_no_time = trainer_no_time.train()
print(f"Final loss (no time embedding): {result_no_time.training_loss:.4f}")

# Test sampling without time embedding
print("\n=== Testing Sampling WITHOUT Time Embeddings ===")
model_no_time.eval()
sampler_no_time = BertSFMSampler(model=model_no_time, tokenizer=tokenizer)
print(f"Input: {test_text}")

for steps in [20, 50]:
    config = BertSFMSamplerConfig(steps=steps, temperature=0.0, prediction_type="endpoint")
    # No time_embedding passed - should work without it
    output = sampler_no_time.infill([inputs["input_ids"][0]], config=config, time_embedding=None)
    output_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Steps={steps:3d}: {output_text}")

print("Training without time embedding test PASSED!")

# Test RK2 integrator
print("\n=== Testing RK2 Integrator ===")
model.eval()
sampler_rk2 = BertSFMSampler(model=model, tokenizer=tokenizer)

print(f"Input: {test_text}")

# Test both Euler and RK2 integrators
for integrator in ["euler", "rk2"]:
    for steps in [20, 40]:
        config = BertSFMSamplerConfig(
            steps=steps,
            temperature=0.0,
            prediction_type="endpoint",
            integrator_type=integrator,
        )
        output = sampler_rk2.infill([inputs["input_ids"][0]], config=config, time_embedding=time_embedding_ce)
        output_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"{integrator:5s} steps={steps:2d}: {output_text}")

print("RK2 integrator test PASSED!")

# Test RK2 with evaluation (prediction_step uses integrator_type from trainer config)
print("\n=== Testing RK2 in Evaluation ===")
model_rk2_eval = transformers.AutoModelForMaskedLM.from_pretrained(model_name)

args_rk2 = BertSFMTrainer.BertSFMConfig(
    output_dir="/tmp/test_bert_sfm_rk2",
    per_device_train_batch_size=20,
    num_train_epochs=5,
    learning_rate=5e-3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    eval_strategy="epoch",
    use_cpu=True,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    loss_type="ce",
    eval_integrator_type="rk2",  # Use RK2 for evaluation
    eval_integration_steps=20,
)

trainer_rk2 = BertSFMTrainer(
    model=model_rk2_eval,
    args=args_rk2,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)

result_rk2 = trainer_rk2.train()
print(f"Final loss with RK2 eval: {result_rk2.training_loss:.4f}")
eval_metrics_rk2 = trainer_rk2.evaluate()
print(f"RK2 eval metrics: {eval_metrics_rk2}")
print("RK2 evaluation test PASSED!")

print("\n=== Test Complete ===")
