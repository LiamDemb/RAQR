import requests


def wikipedia_find_page(title: str):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "RAQRWikipediaLookup/1.0"}
    params = {
        "action": "query",
        "titles": title,
        "format": "json",
        "redirects": 0,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"HTTP error: {e}")
        return False
    except ValueError:
        print("Non-JSON response:")
        print(response.text[:300])
        return False

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return False

    page = next(iter(pages.values()))
    if "missing" in page:
        return False

    found_title = page.get("title")
    return found_title if found_title == title else False
