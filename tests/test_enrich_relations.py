from unittest.mock import patch

from raqr.data.enrich_relations import _triples_for_graph, parse_rebel_output


def test_triples_for_graph_aliased_only_when_add_both_disabled():
    """Without ADD_BOTH_ALIAS_AND_RAW_TRIPLES, returns only aliased triple."""
    with patch("raqr.data.enrich_relations.add_both_alias_and_raw_triples", return_value=False):
        triples = _triples_for_graph("cabaret artist", "occupation", "Eva Busch", {"cabaret artist": "cabaret"})
    assert len(triples) == 1
    assert triples[0].subj_norm == "cabaret"
    assert triples[0].obj_norm == "eva busch"


def test_triples_for_graph_add_both_when_different():
    """With ADD_BOTH_ALIAS_AND_RAW_TRIPLES, returns aliased and raw when they differ."""
    with patch("raqr.data.enrich_relations.add_both_alias_and_raw_triples", return_value=True):
        triples = _triples_for_graph("cabaret artist", "occupation", "Eva Busch", {"cabaret artist": "cabaret"})
    assert len(triples) == 2
    norms = [(t.subj_norm, t.obj_norm) for t in triples]
    assert ("cabaret", "eva busch") in norms
    assert ("cabaret artist", "eva busch") in norms


def test_triples_for_graph_add_both_no_extra_when_same():
    """With ADD_BOTH enabled but alias unchanged, returns only one triple."""
    with patch("raqr.data.enrich_relations.add_both_alias_and_raw_triples", return_value=True):
        triples = _triples_for_graph("singer", "occupation", "Eva Busch", {})
    assert len(triples) == 1
    assert triples[0].subj_norm == "singer"
    assert triples[0].obj_norm == "eva busch"


def test_parse_rebel_output_triplet_format():
    output = "<triplet> Steve Jobs <subj> Apple <obj> founded"
    triples = parse_rebel_output(output)
    assert triples == [("Steve Jobs", "founded", "Apple")]


def test_parse_rebel_output_multiple_triples():
    output = (
        "<triplet> Barack Obama <subj> United States <obj> country of citizenship "
        "<triplet> Barack Obama <subj> Michelle Obama <obj> spouse"
    )
    triples = parse_rebel_output(output)
    assert ("Barack Obama", "country of citizenship", "United States") in triples
    assert ("Barack Obama", "spouse", "Michelle Obama") in triples


def test_parse_rebel_output_strips_special_tokens():
    output = "<triplet> The Walking Dead <subj> eighth season <obj> has part</s>"
    triples = parse_rebel_output(output)
    assert triples == [("The Walking Dead", "has part", "eighth season")]
