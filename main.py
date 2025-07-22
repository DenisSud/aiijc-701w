# Pure math training script (improved)
"""
Reinforcement Learning (RL) stage with better reward functions and comprehensive metrics tracking
"""

import re
from fractions import Fraction
from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import torch
import wandb
from collections import defaultdict
import numpy as np

torch.cuda.empty_cache()

# Configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATA_PATH = "/content/train.csv"
OUTPUT_DIR = "grpo_math_model"
TRAINING_TEMPERATURE = 1.0  # Fixed temperature
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32

# Global metrics tracking
training_metrics = defaultdict(list)

dataset = load_dataset(
    "csv",
    data_files={"train": DATA_PATH},
    split="train"
)

def add_instruction(example):
    example["prompt"] = (
        "Please solve the following problem step by step:\n"
        f"Problem: {example['task'].strip()}\n\n"
        "Show your reasoning inside <thinking> tags, then provide your final numerical answer in brackets like [52].\n"
        "Format: <thinking>your reasoning here</thinking>[answer]"
    )
    return example

dataset = dataset.map(add_instruction)

# Create a lookup dictionary for faster access
dataset_lookup = {example["prompt"]: example for example in dataset}

def parse_answer(s):
    """Extracts answer from various formats, returns float or None."""
    if s is None:
        return None, "none_input"

    s = str(s).strip()

    # Try different answer formats in order of preference
    patterns = [
        (r"\[(\d+(?:\.\d+)?)\]", "bracket_format"),           
        (r"\\boxed\{(\d+(?:\.\d+)?)\}", "latex_boxed"),    
        (r"boxed\{(\d+(?:\.\d+)?)\}", "boxed_format"),      
        (r"answer is (\d+(?:\.\d+)?)", "answer_is"),     
        (r"(\d+(?:\.\d+)?)(?:\s*$|\s*\n)", "number_at_end"), 
    ]

    for pattern, format_type in patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            break
    else:
        return None, "no_match_found"

    try:
        return float(val), format_type
    except ValueError:
        try:
            return float(Fraction(val)), f"{format_type}_fraction"
        except (ValueError, ZeroDivisionError):
            return None, "parse_error"

def check_format(prompts, completions, **kwargs):
    """
    Simple format checker:
    1. Must contain <thinking> tags
    2. Must have numeric answer in brackets [number]
    3. No other characters outside thinking tags and square brackets
    
    Returns 1.0 for correct format, 0.0 for incorrect format.
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        if completion is None or len(completion.strip()) == 0:
            training_metrics["format_empty_completions"].append(1)
            rewards.append(0.0)
            continue
            
        training_metrics["format_empty_completions"].append(0)
        completion = completion.strip()
        
        # Check 1: Must contain <thinking> tags
        has_thinking_open = re.search(r'<thinking>', completion) is not None
        has_thinking_close = re.search(r'</thinking>', completion) is not None
        
        training_metrics["has_thinking_tags"].append(int(has_thinking_open and has_thinking_close))
        
        if not (has_thinking_open and has_thinking_close):
            rewards.append(0.0)
            continue
            
        # Check 2: Must have numeric answer in brackets
        bracket_match = re.search(r'\[(\d+(?:\.\d+)?)\]', completion)
        training_metrics["has_bracket_answer"].append(int(bracket_match is not None))
        
        if not bracket_match:
            rewards.append(0.0)
            continue
            
        # Check 3: Remove thinking tags and bracketed answer, check if anything else remains
        without_thinking = re.sub(r'<thinking>.*?</thinking>', '', completion, flags=re.DOTALL)
        without_answer = re.sub(r'\[\d+(?:\.\d+)?\]', '', without_thinking)
        
        has_extra_content = bool(without_answer.strip())
        training_metrics["has_extra_content"].append(int(has_extra_content))
        
        if has_extra_content:
            rewards.append(0.0)
            continue
            
        # All checks passed
        rewards.append(1.0)
    
    return rewards

def check_answer(prompts, completions, **kwargs):
    """
    Binary accuracy reward: 1.0 if answer matches exactly, 0.0 otherwise.
    Also tracks comprehensive metrics.
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # Get true answer from dataset lookup
        example = dataset_lookup.get(prompt)
        if example is None:
            training_metrics["dataset_lookup_failures"].append(1)
            rewards.append(0.0)
            continue
        
        training_metrics["dataset_lookup_failures"].append(0)
        
        # Track response length metrics
        completion_length = len(completion) if completion else 0
        word_count = len(completion.split()) if completion else 0
        training_metrics["completion_lengths"].append(completion_length)
        training_metrics["completion_word_counts"].append(word_count)
        
        # Track context usage
        prompt_length = len(prompt)
        total_length = prompt_length + completion_length
        training_metrics["prompt_lengths"].append(prompt_length)
        training_metrics["total_sequence_lengths"].append(total_length)
        training_metrics["context_utilization_ratio"].append(total_length / MAX_SEQ_LENGTH)
        
        # Check for potential context overflow
        context_overflow = total_length > MAX_SEQ_LENGTH * 0.95  # 95% threshold
        training_metrics["context_near_overflow"].append(int(context_overflow))
        
        # Parse answers and track parsing success
        true_val, true_parse_type = parse_answer(example["answer"])
        pred_val, pred_parse_type = parse_answer(completion)
        
        training_metrics["true_answer_parse_success"].append(int(true_val is not None))
        training_metrics["pred_answer_parse_success"].append(int(pred_val is not None))
        training_metrics["pred_parse_types"].append(pred_parse_type)
        
        # Track parsing errors
        if true_val is None:
            training_metrics["true_answer_parse_errors"].append(1)
            rewards.append(0.0)
            continue
        
        training_metrics["true_answer_parse_errors"].append(0)
        
        if pred_val is None:
            training_metrics["pred_answer_parse_errors"].append(1)
            rewards.append(0.0)
            continue
            
        training_metrics["pred_answer_parse_errors"].append(0)
        
        # Binary accuracy check
        is_correct = abs(pred_val - true_val) < 1e-6  # Float comparison with tolerance
        training_metrics["answer_accuracy"].append(int(is_correct))
        
        # Track error magnitudes for analysis (even though not used in reward)
        error_magnitude = abs(pred_val - true_val)
        training_metrics["answer_error_magnitudes"].append(error_magnitude)
        
        # Categorize error types
        if error_magnitude == 0:
            training_metrics["error_categories"].append("exact_match")
        elif error_magnitude < 1:
            training_metrics["error_categories"].append("small_error")
        elif error_magnitude < 10:
            training_metrics["error_categories"].append("medium_error")
        else:
            training_metrics["error_categories"].append("large_error")
        
        rewards.append(1.0 if is_correct else 0.0)
    
    return rewards

def log_batch_metrics():
    """Log aggregated metrics to wandb after each batch"""
    if not training_metrics:
        return
        
    # Aggregate metrics
    aggregated = {}
    
    for metric_name, values in training_metrics.items():
        if not values:
            continue
            
        if metric_name in ["pred_parse_types", "error_categories"]:
            # Handle categorical metrics
            from collections import Counter
            counter = Counter(values)
            for category, count in counter.items():
                aggregated[f"{metric_name}_{category}"] = count / len(values)
        else:
            # Handle numerical metrics
            values_array = np.array(values)
            aggregated[f"{metric_name}_mean"] = values_array.mean()
            aggregated[f"{metric_name}_std"] = values_array.std()
            if metric_name in ["completion_lengths", "completion_word_counts", "total_sequence_lengths"]:
                aggregated[f"{metric_name}_max"] = values_array.max()
                aggregated[f"{metric_name}_min"] = values_array.min()
    
    # Log to wandb
    wandb.log(aggregated)
    
    # Clear metrics for next batch
    training_metrics.clear()

# Model setup (unchanged)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = False,
    fast_inference = True,
    gpu_memory_utilization = 0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = LORA_RANK*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Calculate lengths
maximum_length = max(len(row["prompt"]) for row in dataset)
max_prompt_length = maximum_length + 1
max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

vllm_sampling_params = SamplingParams(
    temperature = TRAINING_TEMPERATURE,  # Fixed to 1.0
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 701,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = TRAINING_TEMPERATURE,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 4,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 100,
    save_steps = 100,
    report_to = "wandb",
    output_dir = "outputs",
    fp16_full_eval = True,
    per_device_eval_batch_size = 4,
    eval_accumulation_steps = 1,
    eval_strategy = "steps",
    eval_steps = 1,
)

# Custom trainer class to add metrics logging
class MetricsGRPOTrainer(GRPOTrainer):
    def log_batch_metrics(self):
        log_batch_metrics()

trainer = MetricsGRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        check_format,
        check_answer,
    ],
    args = training_args,
    train_dataset = dataset,
)

# Add callback to log metrics after each batch
class MetricsCallback:
    def on_log(self, args, state, control, logs=None, **kwargs):
        log_batch_metrics()

trainer.add_callback(MetricsCallback())
trainer.train()
