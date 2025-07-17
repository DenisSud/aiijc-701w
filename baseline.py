import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

def get_answer_from_model(task_text):
    sys_prompt = """
    You are a mathematician. You need to solve the task.
    """
    messages = [
        {
            "role": "system",
            "content": sys_prompt,
        },
        {
            "role": "user",
            "content": f"Answer only the final numeric result, nothing else. Task: {task_text}",
        }
    ]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=50)
    result = tokenizer.decode(
        outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
    )

    cleaned = re.findall(r"-?\d+(?:\.\d+)?", result)
    if cleaned:
        extracted_answer = cleaned[0]
    else:
        extracted_answer = result.strip()
    return extracted_answer

df = pd.read_csv("train.csv")

submission_rows = []
for idx, row in df.iterrows():
    task = row["task"]
    answer = get_answer_from_model(task)
    submission_rows.append({"task": task, "answer": f"[{answer}]"})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv("submission.csv", index=False)
