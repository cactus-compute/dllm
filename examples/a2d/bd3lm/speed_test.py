"""
Speed comparison test: BD3LM/MDLM vs AR Qwen3 0.6B

Usage:
    python -u examples/a2d/bd3lm/speed_test.py
    python -u examples/a2d/bd3lm/speed_test.py --diffusion_path "YOUR_MODEL" --ar_path "Qwen/Qwen3-0.6B"
    python -u examples/a2d/bd3lm/speed_test.py --sampler_type mdlm --diffusion_path "dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1"
"""

import time
from dataclasses import dataclass, field

import torch
import transformers

import dllm


@dataclass
class SpeedTestConfig:
    diffusion_path: str = "dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    ar_path: str = "Qwen/Qwen3-0.6B"
    sampler_type: str = "bd3lm"  # "bd3lm" or "mdlm"
    num_samples: int = 100
    max_new_tokens: int = 128
    seed: int = 42
    # Diffusion specific
    steps: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"

    def __post_init__(self):
        self.diffusion_path = dllm.utils.resolve_with_base_env(self.diffusion_path, "BASE_MODELS_DIR")
        self.ar_path = dllm.utils.resolve_with_base_env(self.ar_path, "BASE_MODELS_DIR")


# 100 diverse prompts for testing
DIVERSE_PROMPTS = [
    # Math & reasoning (20)
    "What is 15 + 27?",
    "Calculate 144 divided by 12.",
    "If a train travels 60 mph for 3 hours, how far does it go?",
    "What is the square root of 81?",
    "Solve: 3x + 7 = 22",
    "What is 25% of 200?",
    "How many seconds are in 2 hours?",
    "If 5 apples cost $3, how much do 15 apples cost?",
    "What is 7 factorial?",
    "Convert 2.5 kilometers to meters.",
    "What is the area of a circle with radius 5?",
    "If I flip a coin twice, what's the probability of getting two heads?",
    "What is 2 to the power of 10?",
    "How many edges does a cube have?",
    "What is the sum of angles in a triangle?",
    "Simplify: (4 + 3) × 2 - 5",
    "What is 30% tip on a $45 bill?",
    "How many minutes are in a week?",
    "What is the next prime number after 17?",
    "Convert 68°F to Celsius.",
    # Coding (20)
    "Write a Python function to reverse a string.",
    "How do you create a list in Python?",
    "What is a for loop?",
    "Write code to check if a number is even.",
    "What does 'print()' do in Python?",
    "How do you define a function in JavaScript?",
    "What is an array?",
    "Write a function to find the maximum in a list.",
    "What is recursion?",
    "How do you comment code in Python?",
    "Write code to swap two variables.",
    "What is the difference between == and === in JavaScript?",
    "How do you create a dictionary in Python?",
    "What is a class in programming?",
    "Write a function to count vowels in a string.",
    "What is an if-else statement?",
    "How do you read a file in Python?",
    "What is a while loop?",
    "Write code to check if a string is a palindrome.",
    "What does 'len()' return in Python?",
    # General knowledge (20)
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "What year did World War II end?",
    "What is photosynthesis?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the speed of light?",
    "Who invented the telephone?",
    "What is the tallest mountain on Earth?",
    "What is DNA?",
    "Who was the first person on the moon?",
    "What is the largest ocean?",
    "What causes seasons on Earth?",
    "What is the capital of Japan?",
    "Who discovered gravity?",
    "What is the human body's largest organ?",
    "How many bones are in the adult human body?",
    "What is the Pythagorean theorem?",
    # Creative writing (20)
    "Write a haiku about the ocean.",
    "Describe a sunset in one sentence.",
    "Create a name for a fantasy kingdom.",
    "Write a short joke.",
    "Describe rain using three adjectives.",
    "Write a motivational quote.",
    "Create a metaphor for happiness.",
    "Write a limerick about a cat.",
    "Describe the smell of coffee.",
    "Write a tagline for a pizza restaurant.",
    "Create an alliterative phrase about summer.",
    "Write a simile for speed.",
    "Describe a forest in autumn.",
    "Write a short tongue twister.",
    "Create a name for a sci-fi spaceship.",
    "Write a two-line poem about stars.",
    "Describe the taste of chocolate.",
    "Create a motto for a sports team.",
    "Write an onomatopoeia sentence.",
    "Describe a rainbow to someone who can't see.",
    # Instructions & explanations (20)
    "How do you boil an egg?",
    "Explain what a computer virus is.",
    "How do you tie a shoelace?",
    "What is machine learning in simple terms?",
    "How do you make coffee?",
    "Explain gravity to a child.",
    "How do you change a tire?",
    "What is the internet?",
    "How do plants grow?",
    "Explain what a black hole is.",
    "How do you send an email?",
    "What causes thunder?",
    "How does a bicycle work?",
    "Explain what inflation means.",
    "How do you brush your teeth properly?",
    "What is cryptocurrency?",
    "How do airplanes fly?",
    "Explain what a vaccine does.",
    "How do you save money?",
    "What is climate change?",
]


def format_prompt(prompt: str) -> list[dict]:
    """Format a prompt as a chat message."""
    return [{"role": "user", "content": prompt}]


def benchmark_diffusion(config: SpeedTestConfig) -> dict:
    """Benchmark diffusion model (BD3LM or MDLM)."""
    sampler_type = config.sampler_type.upper()
    print("\n" + "=" * 60)
    print(f"Loading {sampler_type} model...")
    print("=" * 60)

    # Create a simple namespace for model loading
    class ModelArgs:
        def __init__(self, path):
            self.model_name_or_path = path
            self.dtype = "bfloat16"
            self.load_in_4bit = False
            self.attn_implementation = None

    model_args = ModelArgs(config.diffusion_path)
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Select sampler based on type
    if config.sampler_type.lower() == "mdlm":
        sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
        sampler_config = dllm.core.samplers.MDLMSamplerConfig(
            steps=config.steps,
            max_new_tokens=config.max_new_tokens,
            block_size=config.block_size,
            temperature=config.temperature,
            remasking=config.remasking,
        )
    else:
        sampler = dllm.core.samplers.BD3LMSampler(model=model, tokenizer=tokenizer)
        sampler_config = dllm.core.samplers.BD3LMSamplerConfig(
            steps=config.steps,
            max_new_tokens=config.max_new_tokens,
            block_size=config.block_size,
            temperature=config.temperature,
            remasking=config.remasking,
        )

    print(f"Model loaded. Running {config.num_samples} samples...")

    # Warmup
    warmup_messages = [format_prompt(DIVERSE_PROMPTS[0])]
    warmup_inputs = tokenizer.apply_chat_template(
        warmup_messages, add_generation_prompt=True, tokenize=True
    )
    with torch.no_grad():
        _ = sampler.sample(warmup_inputs, sampler_config)

    # Benchmark
    total_tokens = 0
    times = []

    for i, prompt in enumerate(DIVERSE_PROMPTS[: config.num_samples]):
        messages = [format_prompt(prompt)]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        input_len = len(inputs[0])

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            outputs = sampler.sample(inputs, sampler_config, return_dict=True)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        output_len = outputs.sequences.shape[1] - input_len
        total_tokens += output_len
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{config.num_samples}")

    # Cleanup
    del model, sampler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "model": sampler_type,
        "total_time": sum(times),
        "total_tokens": total_tokens,
        "avg_time_per_sample": sum(times) / len(times),
        "tokens_per_second": total_tokens / sum(times),
        "samples": len(times),
    }


def benchmark_ar(config: SpeedTestConfig) -> dict:
    """Benchmark standard AR Qwen3 model."""
    print("\n" + "=" * 60)
    print("Loading AR Qwen3 model...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.ar_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.ar_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Running {config.num_samples} samples...")

    # Warmup
    warmup_messages = [format_prompt(DIVERSE_PROMPTS[0])]
    warmup_inputs = tokenizer.apply_chat_template(
        warmup_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        _ = model.generate(
            warmup_inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Benchmark
    total_tokens = 0
    times = []

    for i, prompt in enumerate(DIVERSE_PROMPTS[: config.num_samples]):
        messages = [format_prompt(prompt)]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        input_len = inputs.shape[1]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        output_len = outputs.shape[1] - input_len
        total_tokens += output_len
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{config.num_samples}")

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "model": "AR Qwen3",
        "total_time": sum(times),
        "total_tokens": total_tokens,
        "avg_time_per_sample": sum(times) / len(times),
        "tokens_per_second": total_tokens / sum(times),
        "samples": len(times),
    }


def print_results(diffusion_results: dict, ar_results: dict):
    """Print comparison results."""
    diffusion_name = diffusion_results["model"]

    print("\n" + "=" * 70)
    print("SPEED TEST RESULTS".center(70))
    print("=" * 70)

    print(f"\n{'Metric':<30} {diffusion_name:>18} {'AR Qwen3':>18}")
    print("-" * 70)
    print(
        f"{'Total time (s)':<30} {diffusion_results['total_time']:>18.2f} {ar_results['total_time']:>18.2f}"
    )
    print(
        f"{'Total tokens generated':<30} {diffusion_results['total_tokens']:>18} {ar_results['total_tokens']:>18}"
    )
    print(
        f"{'Avg time per sample (s)':<30} {diffusion_results['avg_time_per_sample']:>18.3f} {ar_results['avg_time_per_sample']:>18.3f}"
    )
    print(
        f"{'Tokens/second':<30} {diffusion_results['tokens_per_second']:>18.2f} {ar_results['tokens_per_second']:>18.2f}"
    )
    print("-" * 70)

    speedup = ar_results["tokens_per_second"] / diffusion_results["tokens_per_second"]
    if speedup > 1:
        print(f"\nAR Qwen3 is {speedup:.2f}x faster than {diffusion_name}")
    else:
        print(f"\n{diffusion_name} is {1/speedup:.2f}x faster than AR Qwen3")

    print("\n" + "=" * 70)


def main():
    parser = transformers.HfArgumentParser(SpeedTestConfig)
    (config,) = parser.parse_args_into_dataclasses()

    transformers.set_seed(config.seed)

    sampler_type = config.sampler_type.upper()
    print("\n" + "=" * 70)
    print(f"GENERATION SPEED TEST: {sampler_type} vs AR Qwen3 0.6B".center(70))
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  {sampler_type} model: {config.diffusion_path}")
    print(f"  AR model: {config.ar_path}")
    print(f"  Num samples: {config.num_samples}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  {sampler_type} steps: {config.steps}")
    print(f"  {sampler_type} block_size: {config.block_size}")

    # Run benchmarks
    diffusion_results = benchmark_diffusion(config)
    ar_results = benchmark_ar(config)

    # Print comparison
    print_results(diffusion_results, ar_results)


if __name__ == "__main__":
    main()
