# Main training script
"""
Supervised Finetuning (Priming) stage

The model will go through standard instruction fine tuning.
The dataset has been artificially generated using a subset of the original dataset and a larger,
more trust worthy model (deepseek-r1-0528) that generated the reasoning and answers.
"""

# TODO

"""
Reinforcement Learning (RL) stage

Here the model will be trained via GRPO with accuracy based reward functions.
(minimizing l2 distance between predicted and actual values)

Since smaller models generally need more context length to reason to reach the same performance as larger models, we will iteratively increase the context length.
For the first training run the context length will be set to 4096, when the model tries to exceed the context length >2% of the time, we will increase the context length by 4096 and continue training.
We will repeat this pattern until the context length reaches a maximum of 12228 tokens.

This should be approximately ~2000 training steps.
"""

import re
import pandas as pd
import sympy as sp
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import torch

# 1. Load CSV and prepare ground-truth strings
csv_path = "data/train.csv"  # Path to your CSV file
# CSV columns: task, answer (e.g. "[600]", "[-4/23]", "[0.0029]")
df = pd.read_csv(csv_path)
# Strip square brackets from answers
df["answer_str"] = df["answer"].str.strip("[]")

# 2. Build Hugging Face Dataset with only plain Python types
raw_ds = Dataset.from_pandas(
    df[["task", "answer_str"]].rename(columns={"answer_str": "true_str"})
)

# 3. Define the system prompt enforcing <thinking> tags and format
SYSTEM_PROMPT = (
    "You are a chain-of-thought math solver.\n"
    "On each problem, output exactly:\n"
    "  <thinking>\n"
    "    ...your reasoning...\n"
    "  </thinking>\n"
    "Then on the last line output one simplified expression in Python style:\n"
    "  • Integers: 42 or -7\n"
    "  • Decimals: 0.25 or -3.0\n"
    "  • Fractions: -4/23 or 13/7\n\n"
)

def add_prompt(example):
    example["prompt"] = SYSTEM_PROMPT + example["task"]
    return example

# 4. Apply prompt mapping and remove original task column
ds = raw_ds.map(add_prompt, remove_columns=["task"])

# 5. Fixed reward function with correct signature for GRPO
def sympy_full_reward(prompts, completions, **kwargs):
    """
    GRPO reward function that evaluates mathematical expressions using SymPy.
    
    Args:
        prompts: List of prompt strings
        completions: List of completion strings
        **kwargs: Additional keyword arguments
        
    Returns:
        List of reward scores
    """
    rewards = []
    
    # Extract true answers from the dataset during training
    # We need to parse the prompts to get the original task and match with ground truth
    for prompt, completion in zip(prompts, completions):
        # Extract the task from the prompt (remove system prompt)
        task = prompt.replace(SYSTEM_PROMPT, "").strip()
        
        # Find corresponding ground truth from dataset
        # This is a bit inefficient but works for the reward function
        matching_rows = df[df["task"] == task]
        if len(matching_rows) == 0:
            rewards.append(-100.0)
            continue
            
        true_str = matching_rows.iloc[0]["answer_str"]
        
        # Split and clean lines from completion
        lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
        
        # Check format: tags and at least one reasoning line
        if len(lines) < 3 or lines[0] != "<thinking>" or lines[-2] != "</thinking>":
            rewards.append(-100.0)
            continue

        expr_str = lines[-1]
        try:
            # Parse predicted expression
            if re.fullmatch(r"-?\d+/\d+", expr_str):
                pred = sp.Rational(expr_str)
            else:
                pred = sp.simplify(sp.sympify(expr_str))
        except Exception:
            rewards.append(-50.0)
            continue

        # Parse true expression
        try:
            if "/" in true_str:
                true = sp.Rational(true_str)
            else:
                true = sp.simplify(sp.nsimplify(float(true_str)))
        except Exception:
            try:
                true = sp.nsimplify(float(true_str))
            except:
                rewards.append(-100.0)
                continue

        # Exact equivalence check
        try:
            if sp.simplify(pred - true) == 0:
                reward = 10.0
            else:
                dist = float(abs(pred - true))
                reward = max(5.0 - dist, -10.0)  # Cap minimum reward
        except Exception:
            reward = -50.0
            
        rewards.append(reward)
    
    return rewards

# Alternative: More efficient reward function using dataset lookup
def create_reward_function_with_lookup(dataset_df):
    """
    Creates a reward function with precomputed lookup for efficiency.
    """
    # Create lookup dictionary for faster access
    task_to_answer = dict(zip(dataset_df["task"], dataset_df["answer_str"]))
    
    def reward_function(prompts, completions, **kwargs):
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            # Extract task from prompt
            task = prompt.replace(SYSTEM_PROMPT, "").strip()
            
            # Get ground truth
            true_str = task_to_answer.get(task)
            if true_str is None:
                rewards.append(-100.0)
                continue
            
            # Parse completion
            lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
            
            # Check format
            if len(lines) < 3 or lines[0] != "<thinking>" or lines[-2] != "</thinking>":
                rewards.append(-100.0)
                continue

            expr_str = lines[-1]
            try:
                # Parse predicted expression
                if re.fullmatch(r"-?\d+/\d+", expr_str):
                    pred = sp.Rational(expr_str)
                else:
                    pred = sp.simplify(sp.sympify(expr_str))
            except Exception:
                rewards.append(-50.0)
                continue

            # Parse true expression
            try:
                if "/" in true_str:
                    true = sp.Rational(true_str)
                else:
                    true = sp.simplify(sp.nsimplify(float(true_str)))
            except Exception:
                try:
                    true = sp.nsimplify(float(true_str))
                except:
                    rewards.append(-100.0)
                    continue

            # Calculate reward
            try:
                if sp.simplify(pred - true) == 0:
                    reward = 10.0
                else:
                    dist = float(abs(pred - true))
                    reward = max(5.0 - dist, -10.0)
            except Exception:
                reward = -50.0
                
            rewards.append(reward)
        
        return rewards
    
    return reward_function

# 6. Load and wrap the Qwen3-0.6B model with LoRA
max_seq_length = 2048
lora_rank = 32
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=701,
)

# 7. Configure GRPO training with correct prompt column
grpo_args = GRPOConfig(
    output_dir="qwen3_math_grpo_sympy_full",
    logging_steps=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=512 + len(SYSTEM_PROMPT),
    max_completion_length=128,
    num_generations=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    use_vllm=True,
)

# Use the more efficient reward function with lookup
reward_func = create_reward_function_with_lookup(df)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=grpo_args,
    train_dataset=ds,
    reward_funcs=reward_func,
    prompt_column_name="prompt",
)

# 8. Start training
if __name__ == "__main__":
    trainer.train()
