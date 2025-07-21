# Main training script
"""
Supervised Finetuning (Priming) stage

# TODO
The model will go through standard instruction fine tuning.
The dataset has been artificially generated using a subset of the original dataset and a larger,
more trust worthy model (deepseek-r1-0528) that generated the reasoning and answers.
"""

"""
Reinforcement Learning (RL) stage

Here the model will be trained via GRPO with accuracy based reward functions.

TODO:
Since smaller models generally need more context length to reason to reach the same performance as larger models, we will iteratively increase the context length.
For the first training run the context length will be set to 4096, when the model tries to exceed the context length >2% of the time, we will increase the context length by 4096 and continue training.
We will repeat this pattern until the context length reaches a maximum of 12228 tokens.

This should be approximately ~2000 training steps.
"""

# Simple implementation
import re
from fractions import Fraction
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "/content/train.csv"
OUTPUT_DIR = "grpo_math_model"

# 1. Load your dataset from a local CSV
dataset = load_dataset(
    "csv",
    data_files={"train": DATA_PATH},
    split="train"
)

# 2. Preprocess: add reasoning instruction to each task
def add_instruction(example):
    example["prompt"] = (
        f"Problem: {example['task'].strip()}\n\n"
        "Please solve this step by step:\n"
        "1. First, understand what is being asked\n"
        "2. Show your reasoning\n"
        "3. Provide your final answer in brackets like [52]\n\n"
        "Your response should end with your final numerical answer in brackets."
    )
    return example

dataset = dataset.map(add_instruction)

# 3. Define parse and reward functions
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

    if val.lower() in ['answer', 'solution', '']:
        return None

    try:
        return float(val)
    except ValueError:
        try:
            return float(Fraction(val))
        except (ValueError, ZeroDivisionError):
            return None

def reward_func(prompts, completions, **kwargs):
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

# 5. Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 4. Configure GRPO training arguments
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    logging_steps=10,
    temperature= 0.6,
    top_p= 0.95,
)

# 6. Initialize GRPOTrainer and train
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,
)

trainer.train()

print("GRPO training complete!")
