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
# Total: 2 * 2 * 2 * 2 * 2 * 2 = 64 runs
GRID = {
    "learning_rate": [1e-4],
    "embed_type": ["spherical", "simplex"],
    "loss_type": ["ce", "mse"],
    "schedule_type": ["linear", "cosine"],
    "loss_weight_type": ["uniform", "time_weighted"],
    "eval_step_weight_cap": [0.0],
}

# Integrator configurations: (integrator_type, steps)
# euler@40 and rk2@20 have similar compute cost (rk2 does 2 forward passes per step)
INTEGRATOR_CONFIGS = [
    ("euler", 40),
    ("rk2", 20),
]

# Fixed parameters
FIXED_PARAMS = {
    "model_name_or_path": "answerdotai/ModernBERT-base",
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
    cap_str = f"cap{params['eval_step_weight_cap']}" if params['eval_step_weight_cap'] > 0 else "nocap"
    integrator_str = f"{params['eval_integrator_type']}{params['eval_integration_steps']}"
    return f"{params['embed_type']}_{params['loss_type']}_{params['schedule_type']}_{params['loss_weight_type']}_{cap_str}_{integrator_str}_{lr_str}"


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
        "--eval_step_weight_cap", str(params["eval_step_weight_cap"]),
        "--eval_integration_steps", str(params["eval_integration_steps"]),
        "--eval_integrator_type", params["eval_integrator_type"],
        "--output_dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"[{run_idx}/{total_runs}] {run_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            timeout=600,  # 10 minute timeout
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"FAILED: return code {result.returncode}")
            print(f"STDERR:\n{result.stderr[-2000:] if result.stderr else 'None'}")
            return {"error": f"Return code {result.returncode}"}

    except subprocess.TimeoutExpired:
        print("TIMEOUT after 10 minutes")
        return {"error": "Timeout"}
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

    # Get best eval loss (NLL from 20-step integration, comparable across all configs)
    best_eval = min(eval_entries, key=lambda e: e.get("eval_loss", float("inf")))
    best_step = best_eval.get("step", 0)

    # Get train nll at best step (raw loss before weighting, from meter)
    train_nll = None
    if train_nll_entries:
        closest = min(train_nll_entries, key=lambda e: abs(e.get("step", 0) - best_step))
        train_nll = closest.get("nll")

    metrics = {
        "best_eval_loss": best_eval.get("eval_loss"),
        "best_step": best_step,
        "train_nll": train_nll,
    }

    print(f"Done: best_eval_loss={metrics.get('best_eval_loss', 'N/A')}")
    return metrics


def main():
    # Generate all parameter combinations (GRID x INTEGRATOR_CONFIGS)
    keys = list(GRID.keys())
    values = list(GRID.values())
    grid_combinations = list(itertools.product(*values))

    # Total = grid combinations * integrator configs
    total_runs = len(grid_combinations) * len(INTEGRATOR_CONFIGS)

    print(f"Grid combinations: {len(grid_combinations)}")
    print(f"Integrator configs: {len(INTEGRATOR_CONFIGS)}")
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
        "eval_step_weight_cap",
        "eval_integrator_type",
        "eval_integration_steps",
        "best_eval_loss",
        "best_step",
        "train_nll",
        "error",
    ]

    # Write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Run all combinations
    idx = 0
    for combo in grid_combinations:
        for integrator_type, integration_steps in INTEGRATOR_CONFIGS:
            idx += 1
            params = dict(zip(keys, combo))
            params["eval_integrator_type"] = integrator_type
            params["eval_integration_steps"] = integration_steps

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
