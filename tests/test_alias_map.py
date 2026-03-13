from raqr.data.alias_map import build_alias_map_from_redirects, normalize_alias_map
from raqr.data.enrich_entities import norm_entity, normalize_key
from raqr.data.wikipedia_client import WikipediaClient


def test_normalize_alias_map():
    alias_map = normalize_alias_map({"U.S.": "United States"})
    assert alias_map["u s"] == "united states"


def test_norm_entity_uses_alias_map():
    alias_map = normalize_alias_map({"U.S.": "United States"})
    assert norm_entity("U.S.", alias_map) == "united states"


def test_parse_redirects_response():
    data = {
        "query": {
            "pages": {
                "1": {
                    "pageid": 1,
                    "title": "United States",
                    "redirects": [{"title": "USA"}, {"title": "U.S."}],
                }
            }
        }
    }
    redirect_map = WikipediaClient.parse_redirects_response(data)
    assert redirect_map["United States"] == ["USA", "U.S."]


def test_normalize_key_strips_leading_determiner():
    assert normalize_key("the United States") == normalize_key("United States")


def test_build_alias_map_can_disable_curated_aliases():
    class FakeWiki:
        def fetch_redirects(self, titles):
            return {"United States": ["USA"]}

    alias_map = build_alias_map_from_redirects(
        titles=["United States"],
        wiki=FakeWiki(),
        curated_aliases={},
    )
    assert "usa" in alias_map
    assert "u s" not in alias_map


def test_build_alias_map_deterministic_ordering():
    class FakeWiki:
        def fetch_redirects(self, titles):
            # Intentionally unsorted + duplicated redirects.
            if "B" in titles:
                return {"B": ["beta", "alpha", "beta"]}
            return {"A": ["zeta", "eta"]}

    titles = ["B", "A", "B"]
    first = build_alias_map_from_redirects(
        titles=titles, wiki=FakeWiki(), curated_aliases={}
    )
    second = build_alias_map_from_redirects(
        titles=list(reversed(titles)), wiki=FakeWiki(), curated_aliases={}
    )
    assert first == second
