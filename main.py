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

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

print(torch.cuda.is_available())

# ─── Config ─────────────────────────────────────────────────────
MODEL_NAME        = "Qwen/Qwen3-0.6B"
CSV_PATH          = "data/train.csv"
INITIAL_CONTEXT   = 4096
MAX_CONTEXT       = 12228
CONTEXT_STEP      = 4096
TARGET_OVERFLOW   = 0.02
RL_STEPS          = 2000
RL_OUTPUT         = "./rl_model"

# ─── Prepare Data ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()

# ─── Load Tokenizer ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ─── Load Model ─────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

