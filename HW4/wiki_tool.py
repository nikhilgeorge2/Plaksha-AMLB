"""
wiki_tool.py — PROVIDED BY INSTRUCTOR. DO NOT MODIFY.

    from wiki_tool import get_links

    links = get_links("https://en.wikipedia.org/wiki/Cricket")
    # [{"text": "British Empire", "url": "https://en.wikipedia.org/wiki/British_Empire"}, ...]

Notes:
- Results are cached — repeated calls for the same page cost nothing.
- There is a 0.6s rate limit between uncached HTTP requests.
- Only article links are returned (no categories, talk pages, etc.).
"""

import gzip, json, threading, time, urllib.error, urllib.parse, urllib.request
from typing import Dict, List

_API_URL    = "https://en.wikipedia.org/w/api.php"
_BASE_URL   = "https://en.wikipedia.org"
_USER_AGENT = "WikipediaAgentGame/1.0 (university course project; nikhil.george@plaksha.edu.in)"
_DELAY      = 0.6
_MAX_RETRIES = 3

_cache: Dict[str, List[Dict[str, str]]] = {}
_last_request: float = 0.0
_rate_lock = threading.Lock()


def _wait():
    global _last_request
    with _rate_lock:
        elapsed = time.time() - _last_request
        if elapsed < _DELAY:
            time.sleep(_DELAY - elapsed)
        _last_request = time.time()


def _api_get(params: dict) -> dict:
    params["format"] = "json"
    url = _API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent":      _USER_AGENT,
        "Accept-Encoding": "gzip",
    })
    for attempt in range(_MAX_RETRIES):
        _wait()
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
            elapsed = time.time() - t0
            if elapsed > 1.0:
                time.sleep(5.0)  # slow response — back off per bot policy
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                time.sleep(retry_after)
            else:
                raise
    raise urllib.error.HTTPError(url, 429, "Rate limited after retries", {}, None)


def _url_to_title(url: str) -> str:
    """'https://en.wikipedia.org/wiki/World_War_II' → 'World War II'"""
    url = url.split("#")[0].rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if not parsed.path.startswith("/wiki/"):
        raise ValueError(f"Not a Wikipedia article URL: {url!r}")
    return urllib.parse.unquote(parsed.path[6:]).replace("_", " ")


def _title_to_url(title: str) -> str:
    return _BASE_URL + "/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="/:-()'.")


def get_links(url: str) -> List[Dict[str, str]]:
    """
    Return all article links on a Wikipedia page.

        links = get_links("https://en.wikipedia.org/wiki/Cricket")
        # [{"text": "British Empire", "url": "https://en.wikipedia.org/wiki/British_Empire"}, ...]

    Results are cached — repeated calls for the same page make one HTTP request.
    """
    title     = _url_to_title(url)
    cache_key = title.lower()
    if cache_key in _cache:
        return _cache[cache_key]

    data = _api_get({"action": "parse", "page": title, "prop": "links",
                     "redirects": "1", "pllimit": "500"})

    if "error" in data:
        raise urllib.error.HTTPError(url, 400,
            f"Wikipedia API: {data['error'].get('info', 'unknown')}", {}, None)

    canonical = data["parse"]["title"]
    links = [{"text": l["*"], "url": _title_to_url(l["*"])}
             for l in data["parse"]["links"] if l["ns"] == 0]

    _cache[cache_key]         = links
    _cache[canonical.lower()] = links
    return links


def verify_path(urls: List[str]) -> dict:
    """
    Check that every consecutive pair in urls is a real Wikipedia hyperlink.
    Returns {"valid": bool, "steps": int, "start": str, "target": str, "edges": [...]}.
    Used by the scorer — not needed in agent.py.
    """
    if len(urls) < 2:
        return {"valid": False, "steps": 0, "start": "", "target": "", "edges": []}

    edges, all_valid = [], True
    for i in range(len(urls) - 1):
        try:
            from_title  = _url_to_title(urls[i])
            to_title    = _url_to_title(urls[i + 1])
            link_titles = {l["text"].lower() for l in get_links(urls[i])}
            ok          = to_title.lower() in link_titles
        except Exception as e:
            edges.append({"from": urls[i], "to": urls[i + 1], "valid": False, "error": str(e)})
            all_valid = False
            continue

        edges.append({"from": from_title, "to": to_title, "valid": ok,
                      "error": None if ok else f"'{to_title}' not in links of '{from_title}'"})
        if not ok:
            all_valid = False

    return {"valid": all_valid, "steps": len(urls) - 1,
            "start": _url_to_title(urls[0]), "target": _url_to_title(urls[-1]),
            "edges": edges}
