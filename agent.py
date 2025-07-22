# Agentic training script (inspired by deepscalar)
"""
Reinforcement Learning (RL) stage

Here the model will be trained via GRPO with accuracy and tool use based reward functions.

"""

""" Baseline: """
# Simple implementation
import re
from fractions import Fraction
from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch

torch.cuda.empty_cache()

# Configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_PATH = "/content/train.csv"
OUTPUT_DIR = "grpo_math_model"
TEMPERATURE = 0.6



dataset = load_dataset(
    "csv",
    data_files={"train": DATA_PATH},
    split="train"
)

def add_instruction(example):
    example["prompt"] = (
        "jlease solve the following problem step by step:\n"
        f"Problem: {example['task'].strip()}\n\n"
        "1. First, understand what is being asked\n"
        "2. Show your reasoning\n"
        "3. Provide your final answer in brackets like [52]\n\n"
        "Your response should end with your final numerical answer in brackets with no other characters."
    )
    return example

dataset = dataset.map(add_instruction)

def parse_answer(s):
    """Extracts answer from various formats, returns float."""
    if s is None:
        return None

    s = str(s)

    # Try different answer formats in order of preference
    patterns = [
        r"\[(\d+(?:\.\d+)?)\]",           # [52] format
        r"\\boxed\{(\d+(?:\.\d+)?)\}",    # \boxed{52} format
        r"boxed\{(\d+(?:\.\d+)?)\}",      # boxed{52} format
        r"answer is (\d+(?:\.\d+)?)",     # "answer is 52"
        r"(\d+(?:\.\d+)?)(?:\s*$|\s*\n)", # number at end
    ]

    for pattern in patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            break
    else:
        return None

    try:
        return float(val)
    except ValueError:
        try:
            return float(Fraction(val))
        except (ValueError, ZeroDivisionError):
            return None

def check_answer(prompts, completions, **kwargs):
    """Reward based on negative absolute error to true answer."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Extract the answer from the original dataset by finding the example
        # that matches this prompt
        matching_example = None
        for example in dataset:
            if example["prompt"] == prompt:
                matching_example = example
                break

        if matching_example is None:
            rewards.append(-1.0)
            continue

        # Debug: print what we're trying to parse
        print(f"Raw answer from dataset: {repr(matching_example['answer'])}")
        print(f"Completion: {repr(completion[:100])}...")  # First 100 chars

        true_val = parse_answer(matching_example["answer"])
        pred = parse_answer(completion)

        print(f"Parsed true_val: {true_val}, pred: {pred}")

        if pred is None or true_val is None:
            rewards.append(-1.0)
        else:
            rewards.append(-abs(pred - true_val))
    return rewards


max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    gpu_memory_utilization = 0.7, # Reduce if out of memory
    # max_lora_rank = lora_rank,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

# finds the longest prompt in the train.csv
maximum_length = max(len(row["prompt"]) for row in dataset)

max_prompt_length = maximum_length + 1 # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 701,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 1e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 100,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",

    # For optional training + evaluation
    fp16_full_eval = False,
    per_device_eval_batch_size = 4,
    eval_accumulation_steps = 1,
    eval_strategy = "steps",
    eval_steps = 1,
)

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        check_format,
        check_answer,
    ],
    args = training_args,
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)
trainer.train()
