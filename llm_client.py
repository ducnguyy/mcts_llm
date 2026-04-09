import re
import os
import time
from openai import OpenAI

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

_client = None
_call_count = 0  # track total LLM calls for cost measurement


def get_client():
    """Lazy-init the OpenAI client pointing at Ollama."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  
        )
    return _client


def reset_call_count():
    global _call_count
    _call_count = 0

def get_call_count():
    return _call_count

def _chat(messages, temperature=0.7, max_tokens=1024):
    global _call_count
    _call_count += 1
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate(prompt, system="", temperature=0.7):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _chat(messages, temperature=temperature)


def self_evaluate(question, answer):
    query = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Analyze this answer strictly and critically. "
        f"Point out every flaw and every possible imperfection. "
        f"Be very harsh in calculating the grade — never give full marks.\n\n"
        f"Output a score between 1.0 and 10.0.\n"
        f"Response format:\n"
        f"[Analysis] <your analysis>\n"
        f"[Score] <decimal from 1.0 to 10.0>"
    )
    response = _chat(
        [{"role": "user", "content": query}],
        temperature=0.3,  # lower temp for more consistent scoring
    )
    return _parse_score(response)


def get_feedback(question, answer):
    query = (
        f"Question: {question}\n"
        f"Current Answer: {answer}\n\n"
        f"Provide detailed feedback to improve this answer. "
        f"Point out specific flaws, missing elements, or areas for improvement. "
        f"Be constructive but thorough."
    )
    return _chat(
        [{"role": "user", "content": query}],
        temperature=0.7,
    )


def refine(question, answer, feedback):
    query = (
        f"Question: {question}\n\n"
        f"Your previous answer:\n{answer}\n\n"
        f"Feedback received:\n{feedback}\n\n"
        f"Please provide an improved answer that addresses all the feedback. "
        f"Give your complete refined answer."
    )
    return _chat(
        [{"role": "user", "content": query}],
        temperature=0.7,
    )

_score_pattern = re.compile(r'-?\d+\.?\d*')


def _parse_score(response):
    score_section = response.split("Score")[-1] if "Score" in response else response
    numbers = _score_pattern.findall(score_section)

    if not numbers:
        numbers = _score_pattern.findall(response)

    if not numbers:
        return 5.0  # default midpoint 

    score = float(numbers[-1])
    if score < -10 or score > 10:
        score = (score + 100.0) / 200.0 * 9.0 + 1.0

    return max(1.0, min(10.0, score))


def normalize_score(score):
    return (score - 1.0) / 9.0


def grade_quality(rubric, answer):
    query = (
        f"{rubric}\n\n"
        f"Text to grade:\n{answer}\n\n"
        f"Output ONLY a score. Response format:\n"
        f"[Score] <number between 0.0 and 1.0>"
    )
    response = _chat(
        [{"role": "user", "content": query}],
        temperature=0.1,
    )
    # Parse a [0, 1] float from the response
    section = response.split("Score")[-1] if "Score" in response else response
    nums = re.findall(r'\b(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)\b', section)
    if nums:
        try:
            return max(0.0, min(1.0, float(nums[-1])))
        except ValueError:
            pass
    # Fallback
    nums = re.findall(r'-?\d+\.?\d*', response)
    for n in reversed(nums):
        try:
            v = float(n)
            if -100 <= v <= 100:
                return max(0.0, min(1.0, v / 100.0 if abs(v) > 1 else v))
        except ValueError:
            pass
    return 0.0