"""Tests for the LLM-as-judge module."""

from unittest.mock import MagicMock, patch

import pytest

from raqr.llm_judge import LLMJudge, DEFAULT_JUDGE_PROMPT


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("raqr.llm_judge.OpenAI")
def test_judge_returns_2_for_comprehensive(mock_openai_class):
    """judge() returns 2 when the model responds with 2 (comprehensive/correct)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("2")
    mock_openai_class.return_value = mock_client

    judge = LLMJudge(model_id="test-model")
    score = judge.judge(
        question="What is the capital of France?",
        gold_answers=["Paris"],
        predicted_answer="Paris is the capital.",
    )
    assert score == 2


@patch("raqr.llm_judge.OpenAI")
def test_judge_returns_0_for_incorrect(mock_openai_class):
    """judge() returns 0 when the model responds with 0 (incorrect/irrelevant)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("0")
    mock_openai_class.return_value = mock_client

    judge = LLMJudge(model_id="test-model")
    score = judge.judge(
        question="What is the capital of France?",
        gold_answers=["Paris"],
        predicted_answer="London.",
    )
    assert score == 0


@patch("raqr.llm_judge.OpenAI")
def test_judge_returns_1_for_partial(mock_openai_class):
    """judge() returns 1 when the model responds with 1 (partial/superficial)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("1")
    mock_openai_class.return_value = mock_client

    judge = LLMJudge(model_id="test-model")
    score = judge.judge(
        question="Q?",
        gold_answers=["A"],
        predicted_answer="Something related to A but incomplete.",
    )
    assert score == 1


@patch("raqr.llm_judge.OpenAI")
def test_judge_returns_0_for_malformed(mock_openai_class):
    """judge() returns 0 for malformed or empty responses."""
    for content in ["", "Maybe", "I'm not sure", "The answer is ambiguous"]:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(content)
        mock_openai_class.return_value = mock_client

        judge = LLMJudge(model_id="test-model")
        score = judge.judge(
            question="Q?",
            gold_answers=["A"],
            predicted_answer="B",
        )
        assert score == 0, f"Expected 0 for content={content!r}"


@patch("raqr.llm_judge.OpenAI")
def test_judge_parses_digit_from_response(mock_openai_class):
    """judge() extracts 0, 1, or 2 from responses like '2.' or 'Score: 1'."""
    for content, expected in [("2.", 2), ("Score: 1", 1), ("0 - incorrect", 0)]:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(content)
        mock_openai_class.return_value = mock_client

        judge = LLMJudge(model_id="test-model")
        score = judge.judge(question="Q?", gold_answers=["A"], predicted_answer="A")
        assert score == expected, f"Expected {expected} for content={content!r}"


@patch("raqr.llm_judge.OpenAI")
def test_judge_prompt_includes_question_gold_pred_and_scale(mock_openai_class):
    """judge() sends a prompt containing question, gold answers, prediction, and 0-2 scale."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("2")
    mock_openai_class.return_value = mock_client

    judge = LLMJudge(model_id="test-model")
    judge.judge(
        question="What is 2+2?",
        gold_answers=["4", "four"],
        predicted_answer="4",
    )

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 1
    prompt = messages[0]["content"]
    assert "What is 2+2?" in prompt
    assert "4" in prompt or "four" in prompt
    assert "gold" in prompt.lower() or "reference" in prompt.lower()
    assert "0" in prompt and "1" in prompt and "2" in prompt
