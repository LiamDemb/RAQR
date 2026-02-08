from raqr.data.enrich_relations import parse_rebel_output


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
