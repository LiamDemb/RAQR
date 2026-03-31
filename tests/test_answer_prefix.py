import pytest

from raqr.generation.answer_prefix import strip_answer_prefix
from raqr.generation.batch import parse_generation_output


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Paris", "Paris"),
        ("Answer: Paris", "Paris"),
        ("answer: Paris", "Paris"),
        ("ANSWER:  Paris", "Paris"),
        ("1. think\n2. think\nAnswer: March 18, 2018", "March 18, 2018"),
        ("1. x\nanswer: foo", "foo"),
    ],
)
def test_strip_answer_prefix(raw: str, expected: str) -> None:
    assert strip_answer_prefix(raw) == expected


def test_parse_generation_output_uses_case_insensitive_answer() -> None:
    line = {
        "custom_id": "q1_dense",
        "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "1. a\nanswer: Yes"}}]}},
    }
    cid, ans = parse_generation_output(line)
    assert cid == "q1_dense"
    assert ans == "Yes"
