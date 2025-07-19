import pandas as pd
import json
import re
import ast

def extract_answer(response):
    """Extract numerical answer from model response, handling structured format."""
    # First try to extract from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # Fallback to full response
        answer_text = response.lower()

    # Original extraction patterns
    patterns = [
        r"(?:answer is|answer:|final answer:)\s*([+-]?\d+(?:\.\d+)?)",
        r"\\boxed\{([^}]+)\}",
        r"\[([^\]]+)\]"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, answer_text)
        if matches:
            try:
                answer = matches[-1].replace('$', '').replace(',', '')
                return int(answer) if '.' not in answer else float(answer)
            except ValueError:
                continue

    # Fallback: last number in response
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', answer_text.split('\n')[-1])
    if numbers:
        try:
            num = numbers[-1]
            return int(num) if '.' not in num else float(num)
        except ValueError:
            pass

    return None

def create_prompt(task):
    """Create reasoning prompt that enforces structured output."""
    return f"""<|im_start|>system
You are a math expert. Solve this problem step by step.

Structure your response as:
<thinking>
[Your step-by-step reasoning here]
</thinking>

<answer>
[Just the final numerical answer]
</answer>
<|im_end|>
<|im_start|>user
{task}
<|im_end|>
<|im_start|>assistant
"""

def parse_expected_answer(answer_str):
    """Parse expected answer from string."""
    try:
        if answer_str.startswith('[') and answer_str.endswith(']'):
            return ast.literal_eval(answer_str)
        return [answer_str]
    except:
        return [answer_str]

def check_correct(extracted, expected_list):
    """Check if extracted answer matches any expected answer."""
    if extracted is None:
        return False

    for expected in expected_list:
        try:
            if extracted == expected:
                return True
            if isinstance(extracted, (int, float)) and isinstance(expected, (int, float)):
                if abs(extracted - expected) < 1e-6:
                    return True
        except:
            continue
    return False
