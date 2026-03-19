import re
import string
from collections import Counter

from .normalization import normalize_text


def compute_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def compute_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_max_f1(prediction: str, gold_answers: list[str]) -> float:
    """Compute max token-F1 over all gold answers."""
    if not gold_answers:
        return 0.0
    return max(compute_f1(prediction, g) for g in gold_answers)


def compute_max_em(prediction: str, gold_answers: list[str]) -> float:
    """Compute max exact match over all gold answers."""
    if not gold_answers:
        return 0.0
    return max(compute_exact_match(prediction, g) for g in gold_answers)
