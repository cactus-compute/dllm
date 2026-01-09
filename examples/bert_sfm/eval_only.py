"""
Evaluation-only script for BERT-SFM models.

Usage:
    # Single GPU:
    python examples/bert_sfm/eval_only.py \
        --model_name_or_path models/ModernBERT-base-sfm/tulu-3-sft-mixture_and_smoltalk/checkpoint-final \
        --dataset_args data/sft/bert_sfm/tulu-3-sft-mixture_and_smoltalk-1024

    # Multi-GPU:
    accelerate launch --multi_gpu examples/bert_sfm/eval_only.py \
        --model_name_or_path models/ModernBERT-base-sfm/tulu-3-sft-mixture_and_smoltalk/checkpoint-final \
        --dataset_args data/sft/bert_sfm/tulu-3-sft-mixture_and_smoltalk-1024
"""

from dataclasses import dataclass, field

import accelerate
import transformers

import dllm
from dllm.pipelines.bert_sfm import BertSFMTrainer

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/ModernBERT-base-sfm/tulu-3-sft-mixture_and_smoltalk/checkpoint-final"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "data/sft/bert_sfm/tulu-3-sft-mixture_and_smoltalk-1024"
    load_preprocessed_data: bool = True
    max_length: int = 1024


@dataclass
class EvalArguments(BertSFMTrainer.BertSFMConfig):
    output_dir: str = "eval_tmp"
    per_device_eval_batch_size: int = 64
    dataloader_num_workers: int = 4
    do_train: bool = False
    do_eval: bool = True


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )

    accelerate.PartialState().wait_for_everyone()

    trainer = BertSFMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset=dataset.get("test", dataset.get("train")),
        args=eval_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id,
            )
        ),
    )

    metrics = trainer.evaluate()

    if trainer.is_world_process_zero():
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
