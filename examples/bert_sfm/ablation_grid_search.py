#!/usr/bin/env python3
"""
Grid search ablation script for BERT SFM training.

Runs all combinations of:
- learning_rate: 1e-4, 5e-5, 1e-5
- embed_type: spherical, simplex
- loss_type: ce, mse
- schedule_type: linear, cosine
- loss_weight_type: uniform, time_weighted

Total: 3 * 2 * 2 * 2 * 2 = 48 combinations

Results are saved to a CSV file with best and final eval losses.
"""

import csv
import itertools
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


# Grid search parameters
GRID = {
    "learning_rate": [1e-4, 5e-5, 1e-5],
    "embed_type": ["spherical", "simplex"],
    "loss_type": ["ce", "mse"],
    "schedule_type": ["linear", "cosine"],
    "loss_weight_type": ["uniform", "time_weighted"],
}

# Fixed parameters
FIXED_PARAMS = {
    "model_name_or_path": "prajjwal1/bert-medium",
    "dataset_args": "tatsu-lab/alpaca",
    "max_length": 512,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 16,
    "group_by_length": "true",
    "save_only_model": "false",
}

# Output directory base
OUTPUT_BASE = "models/ablations/grid_search"


def get_run_name(params: dict) -> str:
    """Generate a short run name from parameters."""
    lr_str = f"lr{params['learning_rate']:.0e}".replace("-0", "-")
    return f"{params['embed_type']}_{params['loss_type']}_{params['schedule_type']}_{params['loss_weight_type']}_{lr_str}"


def run_training(params: dict, output_dir: str, run_idx: int, total_runs: int) -> dict:
    """Run a single training job and return metrics."""
    run_name = get_run_name(params)

    cmd = [
        "accelerate", "launch", "examples/bert_sfm/sft.py",
        "--model_name_or_path", FIXED_PARAMS["model_name_or_path"],
        "--dataset_args", FIXED_PARAMS["dataset_args"],
        "--max_length", str(FIXED_PARAMS["max_length"]),
        "--num_train_epochs", str(FIXED_PARAMS["num_train_epochs"]),
        "--per_device_train_batch_size", str(FIXED_PARAMS["per_device_train_batch_size"]),
        "--per_device_eval_batch_size", str(FIXED_PARAMS["per_device_eval_batch_size"]),
        "--group_by_length", FIXED_PARAMS["group_by_length"],
        "--save_only_model", FIXED_PARAMS["save_only_model"],
        "--learning_rate", str(params["learning_rate"]),
        "--embed_type", params["embed_type"],
        "--loss_type", params["loss_type"],
        "--schedule_type", params["schedule_type"],
        "--loss_weight_type", params["loss_weight_type"],
        "--output_dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"[{run_idx}/{total_runs}] {run_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"FAILED: return code {result.returncode}")
            return {"error": f"Return code {result.returncode}"}

    except subprocess.TimeoutExpired:
        print("TIMEOUT after 10 minutes")
        return {"error": "Timeout"}
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return {"error": str(e)}

    # Parse trainer_state.json for metrics
    trainer_state_path = Path(output_dir) / "trainer_state.json"
    if not trainer_state_path.exists():
        trainer_state_path = Path(output_dir) / "checkpoint-final" / "trainer_state.json"

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

    # Find eval entries and train entries
    eval_entries = [e for e in log_history if "eval_loss" in e]
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]

    if not eval_entries:
        print("No eval entries found")
        return {"error": "No eval entries"}

    # Get best eval loss and corresponding step
    best_eval_entry = min(eval_entries, key=lambda e: e.get("eval_loss", float("inf")))
    best_eval_step = best_eval_entry.get("step", 0)

    # Get final eval entry
    final_eval_entry = eval_entries[-1]
    final_eval_step = final_eval_entry.get("step", 0)

    # Find train loss at the same steps (closest)
    def get_train_loss_at_step(step):
        if not train_entries:
            return None
        closest = min(train_entries, key=lambda e: abs(e.get("step", 0) - step))
        return closest.get("loss")

    metrics = {
        "best_eval_loss": best_eval_entry.get("eval_loss"),
        "best_eval_nll": best_eval_entry.get("eval_nll"),
        "best_eval_ppl": best_eval_entry.get("eval_ppl"),
        "best_eval_step": best_eval_step,
        "train_loss_at_best_eval": get_train_loss_at_step(best_eval_step),
        "final_eval_loss": final_eval_entry.get("eval_loss"),
        "final_eval_nll": final_eval_entry.get("eval_nll"),
        "final_eval_ppl": final_eval_entry.get("eval_ppl"),
        "final_eval_step": final_eval_step,
        "train_loss_at_final_eval": get_train_loss_at_step(final_eval_step),
    }

    # Also get train nll/ppl if available
    if train_entries:
        last_train = train_entries[-1]
        metrics["final_train_loss"] = last_train.get("loss")
        metrics["final_train_nll"] = last_train.get("nll")
        metrics["final_train_ppl"] = last_train.get("ppl")

    print(f"Done: best_eval_loss={metrics.get('best_eval_loss', 'N/A')}")
    return metrics


def main():
    # Generate all parameter combinations
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = list(itertools.product(*values))
    total_runs = len(combinations)

    print(f"Total combinations: {total_runs}")
    print(f"Estimated time: ~{total_runs * 3} minutes (~{total_runs * 3 / 60:.1f} hours)")

    # Prepare CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"ablation_results_{timestamp}.csv"

    fieldnames = [
        "run_name",
        "learning_rate",
        "embed_type",
        "loss_type",
        "schedule_type",
        "loss_weight_type",
        "best_eval_loss",
        "best_eval_nll",
        "best_eval_ppl",
        "best_eval_step",
        "train_loss_at_best_eval",
        "final_eval_loss",
        "final_eval_nll",
        "final_eval_ppl",
        "final_eval_step",
        "train_loss_at_final_eval",
        "final_train_loss",
        "final_train_nll",
        "final_train_ppl",
        "error",
    ]

    # Write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Run all combinations
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        run_name = get_run_name(params)
        output_dir = os.path.join(OUTPUT_BASE, run_name)

        metrics = run_training(params, output_dir, idx, total_runs)

        row = {
            "run_name": run_name,
            **params,
            **metrics,
        }

        # Append to CSV immediately
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"All {total_runs} runs completed!")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
