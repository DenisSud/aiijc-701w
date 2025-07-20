import pandas as pd
import json
import re
import ast
from typing import Union, List, Optional

def parse_expected_answer(answer_str: str) -> List[Union[int, float, str]]:
    """Parse expected answer from string."""
    try:
        if answer_str.startswith('[') and answer_str.endswith(']'):
            return ast.literal_eval(answer_str)
        return [answer_str]
    except:
        return [answer_str]

def check_correct(extracted: Optional[Union[int, float]], expected_list: List[Union[int, float, str]]) -> bool:
    """Check if extracted answer matches any expected answer."""
    if extracted is None:
        return False

    for expected in expected_list:
        try:
            # Convert expected to number if possible
            if isinstance(expected, str):
                try:
                    expected_num = float(expected)
                    if expected_num.is_integer():
                        expected_num = int(expected_num)
                    expected = expected_num
                except ValueError:
                    continue
            
            if extracted == expected:
                return True
            if isinstance(extracted, (int, float)) and isinstance(expected, (int, float)):
                if abs(extracted - expected) < 1e-6:
                    return True
        except:
            continue
    return False

def validate_ollama_connection(base_url: str = "http://localhost:11434/v1") -> bool:
    """Validate that Ollama is running and accessible."""
    import requests
    try:
        response = requests.get(base_url.replace("/v1", "/api/tags"), timeout=5)
        return response.status_code == 200
    except:
        return False

def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except:
        pass
    return []
