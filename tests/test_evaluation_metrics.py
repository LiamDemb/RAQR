import pytest
from raqr.evaluation.metrics import compute_exact_match, compute_f1
from raqr.evaluation.normalization import normalize_text


def test_normalization_lowercase_and_punctuation():
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("This is... somewhat: complicated, don't you think?") == "this is somewhat complicated dont you think"

def test_normalization_articles():
    assert normalize_text("A cat and the dog") == "cat and dog"
    assert normalize_text("An apple a day") == "apple day"

def test_normalization_whitespace():
    assert normalize_text("  spaces   here  ") == "spaces here"

def test_exact_match():
    # Exact match
    assert compute_exact_match("The cat.", "a cat") == 1.0
    assert compute_exact_match("hello world", "Hello, World!") == 1.0
    
    # Mismatch
    assert compute_exact_match("dog", "cat") == 0.0
    assert compute_exact_match("hello there", "hello world") == 0.0

def test_f1_score_perfect():
    assert compute_f1("The beautiful red car", "a beautiful red car") == 1.0

def test_f1_score_partial():
    # pred: "the beautiful cat" -> "beautiful cat" (2 tokens)
    # gold: "a beautiful dog" -> "beautiful dog" (2 tokens)
    # overlap: "beautiful" (1 token)
    # Precision = 1/2, Recall = 1/2 -> F1 = 1/2
    assert compute_f1("the beautiful cat", "a beautiful dog") == 0.5
    
    # pred: "the big red house" -> "big red house" (3 tokens)
    # gold: "red house" -> "red house" (2 tokens)
    # overlap: "red", "house" (2 tokens)
    # Precision = 2/3, Recall = 2/2 = 1.0 -> F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = (4/3) / (5/3) = 0.8
    assert abs(compute_f1("the big red house", "red house") - 0.8) < 1e-5

def test_f1_score_no_overlap():
    assert compute_f1("black cat", "white dog") == 0.0

def test_f1_score_empty_cases():
    assert compute_f1("", "") == 1.0
    assert compute_f1("the a an", "") == 1.0  # Normalizes to empty
    assert compute_f1("something", "") == 0.0
    assert compute_f1("", "something") == 0.0
