"""GSM8K exact numeric scoring; ambiguous/malformed answers fail closed."""
import json
import re
from decimal import Decimal, InvalidOperation

VERSION = "gsm8k_explicit_answer_decimal_v2"
NUMBER = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")


def numeric(value):
    text = str(value).strip()
    if text.startswith("$"):
        text = text[1:].strip()
    if not NUMBER.fullmatch(text):
        return None
    try:
        result = Decimal(text.replace(",", ""))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def extract_answer(text):
    text = str(text).strip()
    candidates = []
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[pos:])
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "answer" in obj:
            value = obj["answer"]
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                return None
            candidates.append(str(value))
    candidates += re.findall(r'"answer"\s*:\s*"([^"\n]*)"', text)
    candidates += re.findall(r'"answer"\s*:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?)\s*(?=[,}\n])', text)
    candidates += re.findall(r"(?im)^\s*(?:answer|final answer)\s*:\s*([^\n]+)$", text)
    candidates += re.findall(r"####\s*([^\n]+)", text)
    if not candidates and numeric(text) is not None:
        candidates = [text]
    values = [numeric(v) for v in candidates]
    if not values or any(v is None for v in values) or len(set(values)) != 1:
        return None
    return values[0]


def score_item(answer, label, task="gsm8k"):
    if isinstance(label, dict):
        task = label.get("task", task)
        label = label.get("answer", label.get("target"))
    if task.lower() != "gsm8k":
        raise ValueError("No approved scorer for " + str(task))
    target = str(label).split("####")[-1].strip()
    expected, observed = numeric(target), extract_answer(answer)
    if expected is None:
        raise ValueError("Invalid GSM8K gold")
    return float(observed is not None and observed == expected)
