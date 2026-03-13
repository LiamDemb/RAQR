import json

from raqr.data.loaders import load_nq


def test_load_nq_parses_minimal_jsonl(tmp_path):
    path = tmp_path / "nq.jsonl"
    payload = {
        "question_text": "Who wrote The Hobbit?",
        "short_answers": [{"text": "J.R.R. Tolkien"}],
        "context": "The Hobbit is a novel by J.R.R. Tolkien.",
        "document_title": "The Hobbit",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.dataset_source == "nq"
    assert item.gold_answers == ["J.R.R. Tolkien"]


def test_load_nq_parses_nested_document(tmp_path):
    path = tmp_path / "nq_nested.jsonl"
    payload = {
        "id": "123",
        "document": {
            "title": "Google",
            "html": "<html><body><p>Google was founded in 1998.</p></body></html>",
            "tokens": [
                {"token": "<p>", "is_html": True},
                {"token": "Google", "is_html": False},
                {"token": "was", "is_html": False},
                {"token": "founded", "is_html": False},
                {"token": "in", "is_html": False},
                {"token": "1998", "is_html": False},
                {"token": "</p>", "is_html": True},
            ],
        },
        "question": {"text": "who founded google"},
        "annotations": [
            {
                "short_answers": [
                    {"text": "Larry Page"},
                    {"text": "Sergey Brin"},
                ]
            }
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Larry Page", "Sergey Brin"]


def test_load_nq_parses_annotations_dict(tmp_path):
    path = tmp_path / "nq_annotations_dict.jsonl"
    payload = {
        "id": "123",
        "document": {
            "title": "Google",
            "html": "<html><body><p>Google was founded by Larry Page.</p></body></html>",
        },
        "question": {"text": "who founded google"},
        "annotations": {
            "id": "0",
            "short_answers": [{"text": "Larry Page"}],
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(items) == 1
    item = items[0]
    assert item.gold_answers == ["Larry Page"]
