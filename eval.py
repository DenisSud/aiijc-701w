from utils import *
import json
from openai import OpenAI

# Constants
DATASET_PATH = "data/train.csv"
MODEL_NAME = "qwen/qwen3-4b:free"  # Ollama model name
MAX_PROBLEMS = 100
OUTPUT_FILE = f"evals/{MODEL_NAME.split('/')[-1].split(':')[0]}.json"
print(OUTPUT_FILE)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Read your API key from the environment
OPENROUTER_API_KEY = "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"
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set the OPENROUTER_API_KEY environment variable")

# Initialize OpenRouter client (OpenAI-compatible)
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# Load data
df = pd.read_csv(DATASET_PATH)
if MAX_PROBLEMS:
    df = df.head(MAX_PROBLEMS)

print(f"Loaded {len(df)} problems")

# Define structured output schema
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

# Process all problems
results = []
correct = 0

print("Processing problems...")
for i, (_, row) in enumerate(df.iterrows()):
    print(f"Progress: {i}/{len(df)}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a math expert. Solve problems step by step and provide your reasoning and final numerical answer."
                },
                {
                    "role": "user",
                    "content": f"Solve this math problem: {row['task']}"
                }
            ],
            response_format=response_format,
            temperature=0.1,
            max_tokens=4096
        )

        # Parse structured response
        content = response.choices[0].message.content
        response_data = json.loads(content)
        extracted = response_data.get("answer")
        thinking = response_data.get("thinking", "")

    except Exception as e:
        print(f"Error processing problem {i}: {e}")
        extracted = None
        thinking = ""
        response_data = {}

    # Evaluate result
    expected = parse_expected_answer(row['answer'])
    is_correct = check_correct(extracted, expected)

    if is_correct:
        correct += 1

    results.append({
        "task": row['task'],
        "expected": expected,
        "thinking": thinking,
        "extracted": extracted,
        "correct": is_correct,
        "raw_response": response_data
    })

# Save and print results
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

accuracy = correct / len(df)
no_answer = sum(1 for r in results if r['extracted'] is None)

print(f"\nResults:")
print(f"Accuracy: {accuracy:.3f} ({correct}/{len(df)})")
print(f"No answer extracted: {no_answer}/{len(df)}")
print(f"Results saved to: {OUTPUT_FILE}")
