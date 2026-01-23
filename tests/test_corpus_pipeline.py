import json

import pytest

from raqr.data.build_graph import build_graph
from raqr.data.canonical_clean import clean_html_to_structured_doc
from raqr.data.chunking import chunk_blocks
from raqr.data.docstore import DocRecord, DocStore
from raqr.data.alias_map import normalize_alias_map
from raqr.data.enrich_entities import norm_entity
from raqr.data.enrich_years import aggregate_year_fields, extract_years
from raqr.data.entity_lexicon import build_entity_lexicon


def test_clean_html_to_structured_doc_parses_blocks():
    html = """
    <html><body>
      <h1>Title</h1>
      <p>First paragraph.</p>
      <ul><li>Item A</li><li>Item B</li></ul>
      <table><tr><th>Year</th><th>Event</th></tr><tr><td>1999</td><td>Test</td></tr></table>
    </body></html>
    """
    doc = clean_html_to_structured_doc(
        html=html,
        doc_id="doc1",
        title="Title",
        url=None,
        anchors={"outgoing_titles": [], "incoming_stub": []},
        source="wiki",
        dataset_origin="wiki",
    )
    assert len(doc.blocks) >= 3
    assert any(block.block_type == "paragraph" for block in doc.blocks)
    assert any(block.block_type == "list" for block in doc.blocks)
    assert any(block.block_type == "table" for block in doc.blocks)


def test_extract_years_and_aggregate():
    text = "In 1999 the event happened again in 2001-2003."
    years = extract_years(text)
    fields = aggregate_year_fields(years, text, token_count=10)
    assert fields["year_min"] == 1999
    assert fields["year_max"] == 2003
    assert 2001 in fields["years"]


def test_norm_entity_alias():
    alias_map = normalize_alias_map({"u.s.": "united states"})
    assert norm_entity("U.S.", alias_map) == "united states"


def test_chunk_blocks_bounds():
    blocks = [
        type("B", (), {"text": "word " * 300, "section_path": ["Lead"], "block_type": "paragraph"})(),
        type("B", (), {"text": "word " * 300, "section_path": ["Lead"], "block_type": "paragraph"})(),
    ]
    chunks = chunk_blocks(blocks, min_tokens=200, max_tokens=500, overlap_tokens=50)
    assert len(chunks) >= 1
    assert all(chunk.token_count >= 200 for chunk in chunks)


def test_build_graph_nodes():
    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {"entities": [{"norm": "united states", "type": "GPE"}]},
        }
    ]
    graph = build_graph(chunks)
    assert "C:c1" in graph.nodes
    assert "E:united states" in graph.nodes


def test_docstore_cache(tmp_path):
    path = tmp_path / "docstore.sqlite"
    store = DocStore(path.as_posix())
    called = {"count": 0}

    def _fetch():
        called["count"] += 1
        return DocRecord(
            title="Test",
            page_id="1",
            revision_id="10",
            url=None,
            html="<p>Test</p>",
            cleaned_text="Test",
            anchors={},
            source="wiki",
            dataset_origin="wiki",
        )

    record_1 = store.get_or_fetch("title:Test", _fetch)
    record_2 = store.get_or_fetch("title:Test", _fetch)
    assert record_1.title == "Test"
    assert record_2.title == "Test"
    assert called["count"] == 1
    store.close()


def test_build_entity_lexicon():
    chunks = [
        {
            "metadata": {
                "entities": [
                    {"norm": "united states", "surface": "United States", "qid": "Q30"}
                ]
            }
        }
    ]
    df = build_entity_lexicon(chunks)
    assert not df.empty
    assert df.iloc[0]["norm"] == "united states"
