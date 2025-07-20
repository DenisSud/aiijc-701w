import sqlite3
import json
import pandas as pd
from openai import OpenAI

# Constants
TRAIN_DB_PATH = "data/train.db"
EVALS_DB_PATH = "data/evals.db"
MODEL_NAME = "qwen3:1.7b"  # Ollama model name
MAX_PROBLEMS = False

OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Initialize OpenRouter client (OpenAI-compatible)
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

# Structured output schema
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_solution",
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": "Step-by-step reasoning process"
                },
                "answer": {
                    "type": "number",
                    "description": "Final numerical answer"
                }
            },
            "required": ["thinking", "answer"],
            "additionalProperties": False
        }
    }
}

# 1) Load all problems from train.db
train_conn = sqlite3.connect(TRAIN_DB_PATH)
train_cursor = train_conn.cursor()
train_cursor.execute("SELECT id, task, answer FROM tasks")
all_tasks = train_cursor.fetchall()
train_conn.close()

if MAX_PROBLEMS:
    all_tasks = all_tasks[:MAX_PROBLEMS]
print(f"Loaded {len(all_tasks)} problems")

# 2) Connect to evals.db and ensure table exists
evals_conn = sqlite3.connect(EVALS_DB_PATH)
evals_cursor = evals_conn.cursor()
evals_cursor.execute("DROP TABLE IF EXISTS evaluations")
evals_cursor.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
  eval_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  problem_id     INTEGER     NOT NULL,
  model_name     TEXT        NOT NULL,
  run_timestamp  DATETIME    DEFAULT CURRENT_TIMESTAMP,
  thinking       TEXT        NOT NULL,
  extracted      REAL        NULL,
  is_correct     INTEGER     NOT NULL,
  FOREIGN KEY(problem_id) REFERENCES tasks(id)
);
""")
evals_conn.commit()

# 3) Find already-completed problem_ids for this model
evals_cursor.execute(
    "SELECT problem_id FROM evaluations WHERE model_name = ?",
    (MODEL_NAME,)
)
completed_ids = {row[0] for row in evals_cursor.fetchall()}

# 4) Iterate and evaluate
for problem_id, task, right_answer in all_tasks:
    if problem_id in completed_ids:
        continue

    # — normalize the right_answer into a plain number —
    if isinstance(right_answer, str):
        # try JSON‐loading "[600]" → [600]
        try:
            loaded = json.loads(right_answer)
        except json.JSONDecodeError:
            loaded = None

        if isinstance(loaded, list) and loaded:
            right_answer = loaded[0]
        elif isinstance(loaded, (int, float)):
            right_answer = loaded
        else:
            # fallback: regex-extract a number inside brackets
            import re
            m = re.search(r'\[\s*([-\d\.]+)\s*\]', right_answer)
            if m:
                num = m.group(1)
                right_answer = float(num) if '.' in num else int(num)
            else:
                # last fallback: try to parse the whole string
                try:
                    right_answer = float(right_answer)
                    if right_answer.is_integer():
                        right_answer = int(right_answer)
                except ValueError:
                    right_answer = None
    else:
        # already numeric in Python
        right_answer = right_answer

    # 4.1) Send the prompt to the LLM
    prompt = f"""
    You are a mathematician. Solve the following problem step by step, and return your reasoning and final answer in JSON format.

    The JSON should follow this exact structure:
    {{
      "thinking": [your thinking process for solving the problem],
      "answer": [your final numeric answer as a number, not a string]
    }}

    Only return a valid JSON object. Do not include any other text.

    Problem:
    {task}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format
    )

    # 4.2) Parse the LLM output
    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
        thinking = parsed["thinking"]
        extracted = parsed.get("answer", None)
    except (json.JSONDecodeError, KeyError):
        thinking = ""
        extracted = None

    # 4.3) Determine correctness
    is_correct = (extracted == right_answer)

    # 4.4) Insert into evaluations
    evals_cursor.execute(
        """
        INSERT INTO evaluations
          (problem_id, model_name, thinking, extracted, is_correct)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            problem_id,
            MODEL_NAME,
            thinking,
            extracted,
            1 if is_correct else 0,
        )
    )
    evals_conn.commit()
    print(f"The answer is {is_correct}")

# 5) Compute and print summary for this model
evals_cursor.execute(
    """
    SELECT
      COUNT(*) AS total,
      SUM(is_correct) AS correct,
      SUM(CASE WHEN extracted IS NULL THEN 1 ELSE 0 END) AS no_answer
    FROM evaluations
    WHERE model_name = ?
    """,
    (MODEL_NAME,)
)
total, correct, no_answer = evals_cursor.fetchone()
accuracy = correct / total if total > 0 else 0.0

print("\nResults for", MODEL_NAME)
print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
print(f"No answer extracted: {no_answer}/{total}")

evals_conn.close()
