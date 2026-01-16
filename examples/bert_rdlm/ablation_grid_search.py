#!/usr/bin/env python3
"""
Grid search ablation script for BERT RDLM training and evaluation.

Runs combinations of critical hyperparameters to find the best configuration
for high-dimensional vocabulary (BERT ~30k).

Results are saved to a CSV file.
"""

import csv
import itertools
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Grid search parameters
# We focus on parameters that affect the prior, noise schedule, and evaluation.
GRID = {
    "sigma_T": [0.2, 0.4, 0.6],           # Noise scale at t=1
    "eval_temperature": [0.0, 0.5, 1.0],   # Sampling temperature
    "weight_right": [0.2, 1.0],            # Endpoint of training interval [0, weight_right]
    "mix_step_thr": [0.0, 0.5],            # Threshold for mixture prior probability schedule
    "embed_type": ["spherical", "simplex"], # Input embedding mapping
}

# Fixed parameters (adjust for your hardware)
FIXED_PARAMS = {
    "num_processes": 8,                   # Number of GPUs/CPUs
    "model_name_or_path": "answerdotai/ModernBERT-base",
    "dataset_args": "tatsu-lab/alpaca",
    "max_length": 1024,
    "num_train_epochs": 1,                # Fewer epochs for ablation
    "per_device_train_batch_size": 24,
    "per_device_eval_batch_size": 12,
    "learning_rate": "5e-5",
    "gradient_accumulation_steps": 1,
    "eval_integration_steps": 256,         # Moderate steps for faster sweep
    "eval_stochastic": "false",            # ODE is more stable in high-D
    "save_only_model": "true",
    "loss_type": "ce",                     # Standard RDLM cross-entropy
}

# Output directory base
OUTPUT_BASE = "models/ablations/rdlm_grid_search"

def get_run_name(params: dict) -> str:
    """Generate a short run name from parameters."""
    return f"sig{params['sigma_T']}_temp{params['eval_temperature']}_wr{params['weight_right']}_thr{params['mix_step_thr']}_{params['embed_type']}"

def run_training(params: dict, output_dir: str, run_idx: int, total_runs: int) -> dict:
    """Run a single training job and return metrics."""
    run_name = get_run_name(params)
    
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(FIXED_PARAMS["num_processes"]),
        "examples/bert_rdlm/sft.py",
        "--model_name_or_path", FIXED_PARAMS["model_name_or_path"],
        "--dataset_args", FIXED_PARAMS["dataset_args"],
        "--max_length", str(FIXED_PARAMS["max_length"]),
        "--num_train_epochs", str(FIXED_PARAMS["num_train_epochs"]),
        "--per_device_train_batch_size", str(FIXED_PARAMS["per_device_train_batch_size"]),
        "--per_device_eval_batch_size", str(FIXED_PARAMS["per_device_eval_batch_size"]),
        "--learning_rate", FIXED_PARAMS["learning_rate"],
        "--gradient_accumulation_steps", str(FIXED_PARAMS["gradient_accumulation_steps"]),
        "--save_only_model", FIXED_PARAMS["save_only_model"],
        "--sigma_T", str(params["sigma_T"]),
        "--eval_temperature", str(params["eval_temperature"]),
        "--weight_right", str(params["weight_right"]),
        "--mix_step_thr", str(params["mix_step_thr"]),
        "--embed_type", params["embed_type"],
        "--loss_type", FIXED_PARAMS["loss_type"],
        "--eval_integration_steps", str(FIXED_PARAMS["eval_integration_steps"]),
        "--eval_stochastic", FIXED_PARAMS["eval_stochastic"],
        "--output_dir", output_dir,
        "--report_to", "none",            # Disable wandb to avoid spam
    ]

    print(f"\n{'='*60}")
    print(f"[{run_idx}/{total_runs}] {run_name}")
    print(f"{'='*60}")

    try:
        # We use a long timeout because 1 epoch of Alpaca can take a while
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"FAILED: return code {result.returncode}")
            print(f"STDERR:\n{result.stderr[-2000:] if result.stderr else 'None'}")
            return {"error": f"Return code {result.returncode}"}

    except Exception as e:
        print(f"EXCEPTION: {e}")
        return {"error": str(e)}

    # Parse trainer_state.json for metrics
    # Look in multiple locations: root, checkpoint-final, or latest numbered checkpoint
    trainer_state_path = Path(output_dir) / "trainer_state.json"
    if not trainer_state_path.exists():
        trainer_state_path = Path(output_dir) / "checkpoint-final" / "trainer_state.json"
    if not trainer_state_path.exists():
        # Find the highest numbered checkpoint
        checkpoints = sorted(Path(output_dir).glob("checkpoint-[0-9]*"),
                           key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            trainer_state_path = checkpoints[-1] / "trainer_state.json"

    if not trainer_state_path.exists():
        print(f"No trainer_state.json found")
        return {"error": "No trainer_state.json"}

    try:
        with open(trainer_state_path) as f:
            state = json.load(f)
    except Exception as e:
        print(f"JSON parse error: {e}")
        return {"error": f"JSON parse error: {e}"}

    # Extract metrics from log_history
    log_history = state.get("log_history", [])

    eval_entries = [e for e in log_history if "eval_loss" in e]
    # nll logged separately by meter callback (raw loss before weighting)
    train_nll_entries = [e for e in log_history if "nll" in e and "eval_nll" not in e]

    if not eval_entries:
        print("No eval entries found")
        return {"error": "No eval entries"}

    # Get best eval loss (NLL from integration)
    best_eval = min(eval_entries, key=lambda e: e.get("eval_loss", float("inf")))
    best_step = best_eval.get("step", 0)

    # Get train nll at best step (raw loss before weighting, from meter)
    train_nll = None
    if train_nll_entries:
        closest = min(train_nll_entries, key=lambda e: abs(e.get("step", 0) - best_step))
        train_nll = closest.get("nll")

    metrics = {
        "best_eval_loss": best_eval.get("eval_loss"),
        "best_eval_ppl": best_eval.get("eval_ppl", "N/A"),
        "best_step": best_step,
        "train_nll": train_nll,
    }

    print(f"Done: best_eval_loss={metrics.get('best_eval_loss', 'N/A')}, ppl={metrics.get('best_eval_ppl', 'N/A')}")
    return metrics

def main():
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_runs = len(combinations)
    print(f"Total combinations to run: {total_runs}")
    print(f"Results will be base-stored in: {OUTPUT_BASE}")

    # Prepare CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"rdlm_ablation_results_{timestamp}.csv"
    
    fieldnames = ["run_name"] + list(GRID.keys()) + [
        "best_eval_loss",
        "best_eval_ppl",
        "best_step",
        "train_nll",
        "error"
    ]
    
    # Ensure output base exists
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, params in enumerate(combinations, 1):
            run_name = get_run_name(params)
            output_dir = os.path.join(OUTPUT_BASE, run_name)
            
            # Skip if already exists and has trainer_state (optional resume logic)
            # if (Path(output_dir) / "trainer_state.json").exists():
            #    print(f"Skipping {run_name}, already complete.")
            #    continue

            metrics = run_training(params, output_dir, i, total_runs)
            
            row = {"run_name": run_name, **params, **metrics}
            writer.writerow(row)
            f.flush()

    print(f"\nGrid search complete. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
