"""
Interactive chat / sampling script for Fisher-Rao Flow Matching BERT models.

Examples
--------
# Chat mode (multi-turn, chat template)
python -u examples/bert_sfm/chat.py --model_name_or_path "YOUR_MODEL_PATH"

# Raw single-turn sampling
python -u examples/bert_sfm/chat.py --model_name_or_path "YOUR_MODEL_PATH" --chat_template False
"""

import sys
from dataclasses import dataclass

import transformers

import dllm
from dllm.pipelines.bert_sfm import BertSFMSampler, BertSFMSamplerConfig


@dataclass
class ScriptArguments:
    model_name_or_path: str = "models/ModernBERT-base-sfm/alpaca/checkpoint-final"
    seed: int = 42
    chat_template: bool = True
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(BertSFMSamplerConfig):
    steps: int = 100
    max_new_tokens: int = 128
    temperature: float = 0.0
    schedule_type: str = "linear"
    inference_scaling: float = 1.0


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = BertSFMSampler(model=model, tokenizer=tokenizer)

    if script_args.chat_template:
        dllm.utils.multi_turn_chat(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )
    else:
        print("\nSingle-turn sampling (no chat template).")
        dllm.utils.single_turn_sampling(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
