from utils import *
from vllm import LLM, SamplingParams

# Constants
DATASET_PATH = "data/train.csv"
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_PROBLEMS = 100
OUTPUT_FILE = "results.json"

# Load data
df = pd.read_csv(DATASET_PATH)
if MAX_PROBLEMS:
    df = df.head(MAX_PROBLEMS)

print(f"Loaded {len(df)} problems")

# Initialize model
model = LLM(model=MODEL_NAME, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

# Process all problems
prompts = [create_prompt(task) for task in df['task']]
outputs = model.generate(prompts, sampling_params)

# Evaluate results
results = []
correct = 0

for i, (_, row) in enumerate(df.iterrows()):
    response = outputs[i].outputs[0].text.strip()
    expected = parse_expected_answer(row['answer'])
    extracted = extract_answer(response)
    is_correct = check_correct(extracted, expected)

    if is_correct:
        correct += 1

    results.append({
        "task": row['task'],
        "expected": expected,
        "response": response,
        "extracted": extracted,
        "correct": is_correct
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
