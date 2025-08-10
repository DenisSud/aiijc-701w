import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from fractions import Fraction
import sqlite3
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc

# Constants
EVALS_DB_PATH = "data/eval.db"
DATA_DF_PATH = "data/train.csv"

def init_database():
    conn = sqlite3.connect(EVALS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        eval_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id     INTEGER     NOT NULL,
        model_name     TEXT        NOT NULL,
        run_timestamp  DATETIME    DEFAULT CURRENT_TIMESTAMP,
        thinking       TEXT        NOT NULL,
        extracted      REAL        NULL,
        is_correct     INTEGER     NOT NULL
    );
    """)
    conn.commit()
    return conn, cursor


MODELS = {
    'qwen3': 'Qwen/Qwen3-8B',
    'deepseek-qwen': "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    'deepseek-rl': "deepseek-ai/deepseek-math-7b-rl",
}

model_cache = {}
tokenizer_cache = {}

def clear_memory():
    """Clear GPU and RAM memory"""
    for model_id in list(model_cache.keys()):
        unload_model(model_id)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Memory cleared")

def unload_model(model_id: str):
    """Unload model from memory"""
    if model_id in model_cache:
        model = model_cache.pop(model_id)
        del model
        print(f"Unloaded model: {model_id}")
    
    if model_id in tokenizer_cache:
        tokenizer = tokenizer_cache.pop(model_id)
        del tokenizer
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(model_id: str):
    """Load model with memory management"""
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        if free_mem < 2: 
            clear_memory()
    
    if model_id not in model_cache:
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        tokenizer_cache[model_id] = tokenizer
        model_cache[model_id] = model
        
    return tokenizer_cache[model_id], model_cache[model_id]

def extract_answer(text: str):
    """Extract numerical answer from text"""
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    
    if matches:
        try:
            return float(matches[-1])
        except:
            pass
    
    fraction_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if fraction_match:
        numerator, denominator = fraction_match.groups()
        try:
            return float(numerator) / float(denominator)
        except:
            pass
    
    try:
        return float(text)
    except:
        return float('nan')

def get_math_solution(model_id: str, content: str):
    """Get math solution with reasoning and answer"""
    prompt = f"""
    You are a mathematician. Solve the following problem step by step, and return your reasoning and final answer in JSON format.
    The JSON should follow this exact structure:
    {{
      "thinking": "[your thinking process]",
      "answer": [your final numeric answer]
    }}
    Only return a valid JSON object. Do not include any other text.
    Problem:
    {content}
    """
    
    tokenizer, model = load_model(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        num_return_sequences=1
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = full_response.replace(prompt, "").strip()
    
    try:
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        json_str = result[json_start:json_end]
        return json.loads(json_str)
    except:
        return {"thinking": result, "answer": None}

def evaluate(prompts, answers, models, db_cursor, db_conn):
    """Evaluate models with SQLite logging"""
    results = {
        'model': [],
        'accuracy': [],
        'total': len(prompts),
        'correct': [],
        'predictions': {}
    }
    
    problem_ids = list(range(len(prompts)))
    
    for model_name, model_code in models.items():
        clear_memory()
        model_predictions = []
        model_correct = 0
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Processing {model_name}")):
            solution = get_math_solution(model_code, prompt)
            thinking = solution.get("thinking", "")
            raw_answer = solution.get("answer", "")
            
            # Extract numerical answer
            if raw_answer is None:
                prediction = extract_answer(thinking)
            else:
                prediction = extract_answer(str(raw_answer))
            
            if isinstance(prediction, str) or np.isnan(prediction):
                prediction = float('nan')
            
            model_predictions.append(prediction)
            
            is_correct = 0
            if not np.isnan(prediction) and abs(prediction - answers[i]) < 1e-5:
                is_correct = 1
                model_correct += 1
            
            db_cursor.execute(
                """
                INSERT INTO evaluations
                    (problem_id, model_name, thinking, extracted, is_correct)
                VALUES (?, ?, ?, ?, ?)
                """,
                (problem_ids[i], model_name, thinking, prediction, is_correct)
            )
        
        db_conn.commit()
        
        accuracy = model_correct / len(prompts)
        results['model'].append(model_name)
        results['accuracy'].append(accuracy)
        results['correct'].append(model_correct)
        results['predictions'][model_name] = model_predictions
    
    return results

def visualize_results(results):
    """Visualize evaluation results"""
    df = pd.DataFrame({
        'Model': results['model'],
        'Accuracy': results['accuracy'],
        'Correct': results['correct'],
        'Total': results['total']
    })
    
    df = df.sort_values('Accuracy', ascending=False)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    ax = sns.barplot(x='Accuracy', y='Model', data=df, palette='viridis')
    plt.title('Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.ylabel('Model')
    
    for i, v in enumerate(df['Accuracy']):
        ax.text(v + 0.01, i, f"{v:.4f}", color='black', va='center')

    plt.subplot(1, 2, 2)
    ax = sns.barplot(x='Correct', y='Model', data=df, palette='rocket')
    plt.title('Correct Answers Count')
    plt.xlabel('Correct Answers')
    plt.ylabel('')
    plt.tight_layout()
    
    for i, v in enumerate(df['Correct']):
        ax.text(v + 0.5, i, f"{v}/{results['total']}", color='black', va='center')
    
    plt.savefig('results_visualization.png')
    plt.show()
    
    return df

def print_summary(db_cursor, models):
    """Print summary from database"""
    for model_name in models.keys():
        db_cursor.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(is_correct) AS correct,
                SUM(CASE WHEN extracted IS NULL THEN 1 ELSE 0 END) AS no_answer
            FROM evaluations
            WHERE model_name = ?
            """,
            (model_name,)
        )
        total, correct, no_answer = db_cursor.fetchone()
        accuracy = correct / total if total > 0 else 0.0

        print("\nResults for", model_name)
        print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"No answer extracted: {no_answer}/{total}")


if __name__ == "__main__":
    db_conn, db_cursor = init_database()
    df = pd.read_csv(DATA_DF_PATH)
    prompts = df['task'][:500]
    answers = df['answer'][:500]
    answers = [float(Fraction(i.replace('[', '').replace(']', ''))) for i in answers]
    
    try:
        results = evaluate(prompts, answers, MODELS, db_cursor, db_conn)
        df_results = visualize_results(results)
        print(df_results)
        
        df_results.to_csv('model_results.csv', index=False)
        
        print_summary(db_cursor, MODELS)
        
    finally:
        # Clean up
        clear_memory()
        db_conn.close()
        print("Database connection closed")