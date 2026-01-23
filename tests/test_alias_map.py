from raqr.data.alias_map import build_alias_map_from_redirects, normalize_alias_map
from raqr.data.enrich_entities import norm_entity
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
