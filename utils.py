from openai import OpenAI, RateLimitError
import re
from fractions import Fraction
import os
from dotenv import load_dotenv
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


env_path = './.env'

try:
    load_dotenv(env_path)
    API_KEY = os.environ["OPENROUTER_API_KEY"]
except KeyError:
    raise ValueError("OPENROUTER_API_KEY не найден в .env")
except Exception as e:
    raise Exception(f"Ошибка при загрузке .env: {e}")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError)
)
def get_answer_from_api(model_name: str, content: str, max_retries=5):
    prompt = (
        f"{content}\n\n"
        "Provide ONLY the final numerical answer without any additional text, explanations or formatting. "
        "If the solution is an equation, give ONLY the numerical value. "
        "Example: If the answer is -5, output ONLY: -5"
    )
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_body={},
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                timeout=30
            )
            result = completion.choices[0].message.content
            return result
        except (json.JSONDecodeError, ConnectionError, TimeoutError) as e:
            wait_time = 2 ** attempt
            print(f"Ошибка ({type(e).__name__}) на попытке {attempt+1}/{max_retries}: {e}. Жду {wait_time} сек...")
            print(e)
            time.sleep(wait_time)
        except Exception as e:
            print(f"Критическая ошибка: {type(e).__name__} - {e}")
            return f"ERROR: {type(e).__name__}"
    
    print(f"Не удалось получить ответ после {max_retries} попыток")
    return "ERROR: MAX_RETRIES_EXCEEDED"


def extract_answer(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    
    if matches:
        return matches[-1]
    
    fraction_match = re.search(r"(\d+)/(\d+)", text) #дробь a/b
    if fraction_match:
        numerator, denominator = fraction_match.groups()
        return float(Fraction(f"{numerator}/{denominator}"))
    
    return float(text)


# Модели с Hugging Face
MODELS = {
    'deepseek': "deepseek-ai/deepseek-math-7b-base",
}

model_cache = {}
tokenizer_cache = {}

def load_model(model_id: str):

    if model_id not in model_cache:
        print(f"Загрузка модели и токенизатора для: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
            device_map="auto"    
        )
        
        tokenizer_cache[model_id] = tokenizer
        model_cache[model_id] = model
        
    return tokenizer_cache[model_id], model_cache[model_id]

def get_local_answer(model_id: str, content: str, max_retries=3):
    prompt = (
        f"{content}\n\n"
        "Provide ONLY the final numerical answer without any additional text, explanations or formatting. "
        "If the solution is an equation, give ONLY the numerical value. "
        "Example: If the answer is -5, output ONLY: -5\n"
        "Answer: "
    )
    
    for attempt in range(max_retries):
        try:
            tokenizer, model = load_model(model_id)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,           
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1,                  
                num_return_sequences=1
            )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = full_response.replace(prompt, "").strip()
            
            return result
                
        except Exception as e:
            print(f"Error: {type(e).__name__} - {e}")
            return f"ERROR: {type(e).__name__}"
    
    print(f"Не удалось получить ответ от модели {model_id} после {max_retries} попыток.")
    return "ERROR: MAX_RETRIES_EXCEEDED"