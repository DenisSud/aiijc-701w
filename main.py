# %% [code] {"execution":{"iopub.status.busy":"2025-08-28T09:49:54.059497Z","iopub.execute_input":"2025-08-28T09:49:54.059780Z"}}
# %%capture
# !pip install --upgrade -qqq uv
# try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
# except: get_numpy = "numpy"
# try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
# except: is_t4 = False
# get_vllm, get_triton = ("vllm==0.10.1", "triton==3.2.0") if is_t4 else ("vllm", "triton")
# !uv pip install -qqq --upgrade     unsloth {get_vllm} {get_numpy} torchvision bitsandbytes xformers
# !uv pip install -qqq {get_triton}
# !uv pip install "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.
# !uv pip install transformers==4.55.4

# %% [markdown] {"id":"ZkH_y8UC9lvv"}
# ### Unsloth

# %% [markdown] {"id":"jN75nmdx9lvw"}
# Goal: To convert `Qwen3-4B-Base` into a reasoning model via GRPO by using OpenR1's Math dataset.
# 
# We first pre fine-tune the model to make GRPO skip trying to match formatting - this speeds GRPO up.

# %% [code] {"id":"yM94yK5zqd1T"}
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
DATA_PATH = "/kaggle/input/aiijc-llm-teacher/train.csv"
OUTPUT_DIR = "/kaggle/working/gspo_math_model/"
SEED = 701
MAX_SEQ_LENGTH = 8192

# %% [code] {"id":"DkIvEkIIkEyB","outputId":"97ea6f41-a359-4cf9-d153-aefa7e358a67"}
from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4, # Reduce if out of memory
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
    random_state = SEED,
)

# %% [code] {"id":"YWSZ0DET7bob","outputId":"a7305d61-dc0d-42e1-e3f4-2d52c059f927"}
import gc
gc.collect()

# %% [markdown] {"id":"7KGgPgk_5S8r"}
# ### Data Prep
# <a name="Data"></a>

# %% [code] {"id":"9MlkSX-0LtRq"}
# =========================
# Prompt Construction
# =========================
SYSTEM_MSG = (
    "You are a helpful mathematical reasoning assistant. "
    "Solve the problem step by step. "
    "At the end, output the final numeric answer in square brackets on a new line, e.g. [600]. "
    "Acceptable final formats: integers, decimals, or simplified fractions like [3/5]."
)

def build_prompt(question: str) -> str:
    # Simple, consistent chat-like tags (not required to be tokenizer special tokens)
    return (
        f"<|system|>\n{SYSTEM_MSG}\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\n"
    )

def normalize_gold_answer(ans_raw: str) -> str:
    # Ground truth answers come in square brackets like "[600]"; keep that format
    return ans_raw.strip()


# %% [code] {"id":"c7Nx9BsOLuEk","outputId":"7ebd9c3b-a837-43df-fbca-0f09a2b75e54"}
from datasets import load_dataset
# =========================
# Load + Preprocess Dataset
# =========================
raw_ds = load_dataset("csv", data_files={"train": DATA_PATH}, split="train")

# Map into fields needed by GRPOTrainer: "prompt" for generation, plus "gold_answer" for rewards
proc_ds = raw_ds.map(
    lambda ex: {
        "prompt": build_prompt(ex["task"]),
        "gold_answer": normalize_gold_answer(ex["answer"]),
    },
    remove_columns=raw_ds.column_names,
)


# %% [code] {"id":"L7vphWGMLvUA","outputId":"5e00a6af-d9ab-4c63-add2-6c891eaac68e"}
import numpy as np
# =========================
# Compute max_len and lengths for GRPO
# =========================
# Tokenize prompts and compute the empirical max token length
def _len_fn(batch):
    toks = tokenizer(
        batch["prompt"],
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    lengths = [len(i) for i in toks["input_ids"]]
    return {"prompt_len": lengths}

proc_ds = proc_ds.map(_len_fn, batched=True, batch_size=256)

# Model/tokenizer context limit
model_max_len = getattr(tokenizer, "model_max_length", None)
if model_max_len is None or model_max_len <= 0 or model_max_len > 10**9:
    model_max_len = MAX_SEQ_LENGTH

# Empirical maximum prompt length in tokens
max_len = int(np.max(proc_ds["prompt_len"]))

# Headroom for EOS or slight tokenization variance
SAFETY = 1
max_prompt_length = min(max_len + SAFETY, model_max_len)

# Choose completion budget with a minimum room
MIN_COMP = 64
max_completion_length = max(1, min(model_max_len - max_prompt_length, model_max_len, MAX_SEQ_LENGTH - max_prompt_length))
if max_completion_length < MIN_COMP:
    # Force room for completion if prompts nearly fill the context
    max_prompt_length = max(model_max_len - MIN_COMP, 1)
    max_completion_length = MIN_COMP

print("Computed lengths:")
print(" - max_len (empirical prompt tokens):", max_len)
print(" - max_prompt_length (used):", max_prompt_length)
print(" - max_completion_length (used):", max_completion_length)
print(" - tokenizer.model_max_length:", model_max_len)


# %% [code] {"id":"whozaAFhJOMD"}
from vllm import SamplingParams
# =========================
# vLLM Sampling Params
# =========================
stop_tokens = [tokenizer.eos_token]

vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = SEED,
    stop = stop_tokens,
    include_stop_str_in_output = True,  # Keep EOS in the captured text for reward parsing safety
    temperature = 1.0,
)



# %% [code] {"id":"hisv11WaLlw0"}
from fractions import Fraction
from typing import Optional, Tuple, List
# =========================
# Reward Function
# =========================
BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")

def extract_final_bracketed(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = BRACKET_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()

def parse_numeric(s: str) -> Optional[Tuple[str, Fraction]]:
    if s is None:
        return None
    s = s.strip()

    # Fraction a/b
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            num, den = parts[0].strip(), parts[1].strip()
            # Try integer fraction
            try:
                f = Fraction(int(num), int(den))
                return ("frac", f)
            except Exception:
                # Fall back to float -> Fraction
                try:
                    f = Fraction(float(num)) / Fraction(float(den))
                    return ("frac", f.limit_denominator(10_000_000))
                except Exception:
                    return None

    # Integer
    try:
        i = int(s)
        return ("int", Fraction(i, 1))
    except Exception:
        pass

    # Float / scientific notation
    try:
        val = float(s)
        return ("float", Fraction(val).limit_denominator(10_000_000))
    except Exception:
        return None

def answers_equal(pred: Fraction, gold: Fraction, tol: float = 1e-6) -> bool:
    if pred == gold:
        return True
    return abs(float(pred) - float(gold)) <= tol

def reward_func(
    completions: List[str],
    prompts: List[str] = None,
    completion_ids: List[List[int]] = None,
    gold_answer: List[str] = None,
    **kwargs,
):
    """
    Unsloth/TRL GRPO reward function signature:
      - completions: list[str]
      - prompts: list[str] (optional)
      - completion_ids: token IDs for completions (optional)
      - any dataset columns are forwarded as keyword args (e.g., gold_answer)

    Returns:
      - list[float] or np.ndarray of shape (batch_size,)
    """
    # Defensive defaults
    if completions is None:
        return np.zeros(1, dtype=np.float32)

    # gold_answer must be provided by your dataset mapping
    if gold_answer is None:
        # If the column name differs, fetch it from kwargs here
        # gold_answer = kwargs.get("labels") or kwargs.get("answers")
        return np.zeros(len(completions), dtype=np.float32)

    rewards = []
    for comp, gold in zip(completions, gold_answer):
        pred_inner = extract_final_bracketed(comp)
        gold_inner = extract_final_bracketed(gold)

        if pred_inner is None or gold_inner is None:
            rewards.append(0.0)
            continue

        pred_num = parse_numeric(pred_inner)
        gold_num = parse_numeric(gold_inner)

        if pred_num is None or gold_num is None:
            rewards.append(0.0)
            continue

        _, pred_frac = pred_num
        _, gold_frac = gold_num

        correct = answers_equal(pred_frac, gold_frac, tol=1e-6)
        rewards.append(1.0 if correct else 0.0)

    return np.array(rewards, dtype=np.float32)

# %% [code] {"id":"C-W8PIB3Lj0n"}

# =========================
# Split Dataset (optional eval)
# =========================
new_dataset = proc_ds.train_test_split(test_size=0.01, seed=SEED)


# %% [code] {"id":"VGBY4laTLihK","outputId":"417efd77-06a5-4702-a426-9b54d8375a30"}
from trl import GRPOConfig, GRPOTrainer

# =========================
# TRL GRPO/GSPO Config
# =========================
training_args = GRPOConfig(
    importance_sampling_level = "sequence",  # GSPO flavor
    vllm_sampling_params = vllm_sampling_params,

    # Optimization
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",

    # Logging/steps
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,  # Increase (e.g., 4) for smoother optimization
    num_generations = 4,              # Fewer if OOM
    num_train_epochs = 1,
    save_steps = 100,
    report_to = "none",
    output_dir = OUTPUT_DIR,

    # Lengths
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,

    # Optional evaluation
    fp16_full_eval = True,
    per_device_eval_batch_size = 4,
    eval_accumulation_steps = 1,
    eval_strategy = "steps",
    eval_steps = 1,
    seed = SEED,
)

# %% [code] {"id":"VCj35RhiLhbf","outputId":"0081bb7b-69dc-4e70-88a5-6abf5f086116"}

# =========================
# Build and Run Trainer
# =========================
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [reward_func],
    args = training_args,

    # Use preprocessed columns
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],

    # Tell TRL which column contains prompts
    prompt_column = "prompt",
)

print("Starting training...")
trainer.train()
print("Training complete. Check outputs at:", OUTPUT_DIR)

# %% [markdown] {"id":"tlaUdxC_VHpz"}
# <a name="Inference"></a>
# ### Inference
# Now let's try the model we just trained! First, let's first try the model without any GRPO trained:

# %% [code] {"id":"qtcz_lpbVC92"}
text = "What is the sqrt of 101?"

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

output

# %% [markdown] {"id":"Colxz9TAVMsi"}
# And now with the LoRA we just trained with GRPO - we first save the LoRA first!

# %% [code] {"id":"AL-BcuB1VLIv"}
model.save_lora("/content/drive/MyDrive/lora_math")

# %% [markdown] {"id":"a4LMOBl8boGX"}
# Verify LoRA is actually trained!

# %% [code] {"id":"4SfdI-ERbpiw"}
from safetensors import safe_open

tensors = {}
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
    # Verify both A and B are non zero
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert(n_zeros.item() != tensor.numel())

# %% [markdown] {"id":"CwpbwnDBVRLg"}
# Now we load the LoRA and test:

# %% [code] {"id":"zf_OY5WMVOxF"}
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "What is the sqrt of 101?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    tokenize = False,
)
from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 2048,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

output

# %% [markdown] {"id":"6aDgFfhFYIAS"}
# Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

# %% [markdown] {"id":"-NUEmHFSYNTp"}
# <a name="Save"></a>
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# %% [code] {"id":"NjXGTkp7YNtB"}
# Merge to 16bit
if True: model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "merged_16bit",)
if True: model.push_to_hub_merged("densud2/Qwen3-1.7B-Math", tokenizer, save_method = "merged_16bit")

# Merge to 4bit
if False: model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "merged_4bit",)
if True: model.push_to_hub_merged("densud2/Qwen3-1.7B-Math", tokenizer, save_method = "merged_4bit")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")


# %% [markdown] {"id":"52WMb3k_YPt8"}
# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# %% [code] {"id":"QyEjW-WuYQIm"}
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )

# %% [markdown] {"id":"V15Yhj1V9lwG"}
# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
