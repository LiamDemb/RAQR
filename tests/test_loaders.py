import json

from raqr.data.loaders import load_complextempqa, load_nq, load_wikiwhy


def test_load_nq_parses_minimal_jsonl(tmp_path):
    path = tmp_path / "nq.jsonl"
    payload = {
        "question_text": "Who wrote The Hobbit?",
        "short_answers": [{"text": "J.R.R. Tolkien"}],
        "context": "The Hobbit is a novel by J.R.R. Tolkien.",
        "document_title": "The Hobbit",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    records = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(records) == 1
    record = records[0]
    assert record.benchmark_item.dataset_source == "nq"
    assert record.benchmark_item.gold_answers == ["J.R.R. Tolkien"]
    assert record.document.content.startswith("The Hobbit")


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
    records = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(records) == 1
    record = records[0]
    assert record.benchmark_item.gold_answers == ["Larry Page", "Sergey Brin"]
    assert "Google was founded" in record.document.content


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
    records = list(load_nq(str(path), dataset_version="v1", max_rows=5))
    assert len(records) == 1
    record = records[0]
    assert record.benchmark_item.gold_answers == ["Larry Page"]


def test_load_complextempqa_parses_json(tmp_path):
    path = tmp_path / "complextempqa.json"
    payload = [
        {
            "question": "When did Apollo 11 land on the Moon?",
            "answers": ["1969"],
            "title": "Apollo 11",
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = list(load_complextempqa(str(path), dataset_version="v1", max_rows=5))
    assert len(records) == 1
    record = records[0]
    assert record.benchmark_item.dataset_source == "complextempqa"
    assert record.benchmark_item.gold_answers == ["1969"]
    assert record.document.metadata.get("context_fallback") is True


def test_load_wikiwhy_parses_csv(tmp_path):
    path = tmp_path / "wikiwhy.csv"
    path.write_text(
        "question,answer,context\n"
        "Why do leaves change color?,Because chlorophyll breaks down,"
        "Leaves change color when chlorophyll breaks down.\n",
        encoding="utf-8",
    )
    records = list(load_wikiwhy(str(path), dataset_version="v1", max_rows=5))
    assert len(records) == 1
    record = records[0]
    assert record.benchmark_item.dataset_source == "wikiwhy"
    assert record.benchmark_item.gold_answers == ["Because chlorophyll breaks down"]


def test_load_wikiwhy_parses_jsonl(tmp_path):
    path = tmp_path / "wikiwhy.jsonl"
    payload = {
        "ctx": "Leaves change color when chlorophyll breaks down.",
        "title": "Leaf color",
        "question": "Why do leaves change color?",
        "cause": "Because chlorophyll breaks down.",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    records = list(load_wikiwhy(str(path), dataset_version="v1", max_rows=5))
    assert len(records) == 1
    record = records[0]
    assert record.benchmark_item.dataset_source == "wikiwhy"
