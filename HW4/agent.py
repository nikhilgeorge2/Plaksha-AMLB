"""
agent.py — submission file.

Improved strategy:
- Strong local heuristics first (no LLM cost).
- Optional LLM reranking over a small candidate set.
- Hard time budgeting across all pairs.
- Link caching reuse across pairs.
- Small bounded BFS "burst" when greedy stalls.

The scorer calls solve_all(pairs, deadline).
"""

import json
import os
import random
import re
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Set

from wiki_tool import get_links

# ── API key — read from file, never hardcode ──────────────────────────────────
_API_KEY_PATH = os.path.expanduser("~/gemini_api_key.txt")
with open(_API_KEY_PATH) as f:
    _API_KEY = f.read().strip()
if not _API_KEY:
    raise RuntimeError("~/gemini_api_key.txt is empty. Put your Gemini key on one line.")

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

# ── Global caches (shared across all pairs in one run) ───────────────────────
_LINK_CACHE: Dict[str, List[Dict[str, str]]] = {}
_TITLE_TOKEN_CACHE: Dict[str, Set[str]] = {}

# Stopwords for rough lexical similarity
_STOP = {
    "the", "of", "and", "in", "to", "a", "an", "on", "for", "by", "with",
    "from", "at", "is", "as", "or", "new", "list", "history", "united",
}


def _ask_gemini(prompt: str) -> str:
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(
        _GEMINI_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=14) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _url_title(url: str) -> str:
    return url.split("/wiki/")[-1].split("#")[0].replace("_", " ").strip()


def _tok(title: str) -> Set[str]:
    k = title.lower()
    if k in _TITLE_TOKEN_CACHE:
        return _TITLE_TOKEN_CACHE[k]
    t = {
        w
        for w in re.findall(r"[a-z0-9]+", k)
        if len(w) > 2 and w not in _STOP
    }
    _TITLE_TOKEN_CACHE[k] = t
    return t


def _sim(a: str, b: str) -> float:
    sa, sb = _tok(a), _tok(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / max(1, len(sb)) + 0.35 * inter / max(1, len(sa))


def _get_links_cached(url: str) -> List[Dict[str, str]]:
    if url in _LINK_CACHE:
        return _LINK_CACHE[url]
    links = get_links(url)
    _LINK_CACHE[url] = links
    return links


def _empty_result(pair: dict) -> dict:
    s = pair["start"]
    return {
        "pair_id": pair["pair_id"],
        "path": [s, s],
        "steps": 1,
        "llm_calls": 0,
        "success": False,
        "link_counts": [],
    }


def _result(pair_id: str, path: List[str], llm_calls: int, link_counts: List[int], success: bool) -> dict:
    return {
        "pair_id": pair_id,
        "path": path,
        "steps": len(path) - 1,
        "llm_calls": llm_calls,
        "success": success,
        "link_counts": link_counts,
    }


def _rank_candidates(
    links: Sequence[Dict[str, str]],
    target_title: str,
    visited: Set[str],
    k: int = 14,
) -> List[Dict[str, str]]:
    scored = []
    for l in links:
        u, t = l["url"], l["text"]
        if u in visited:
            continue
        score = _sim(t, target_title)
        if t.lower() in target_title.lower() or target_title.lower() in t.lower():
            score += 1.4
        scored.append((score, l))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:k]]


def _llm_pick(
    current_title: str,
    target_title: str,
    candidates: Sequence[Dict[str, str]],
) -> Optional[int]:
    if not candidates:
        return None
    numbered = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        "You are navigating Wikipedia by clicking ONE hyperlink.\n"
        f"Current page: {current_title}\n"
        f"Goal page: {target_title}\n\n"
        "Choose the single most promising option to reach the goal quickly.\n"
        "Reply with ONLY one integer index.\n\n"
        f"Options:\n{numbered}"
    )
    out = _ask_gemini(prompt).strip()
    m = re.search(r"\d+", out)
    if not m:
        return None
    idx = int(m.group(0)) - 1
    if 0 <= idx < len(candidates):
        return idx
    return None


def _two_hop_best(current_url: str, target_url: str, visited: Set[str], budget_expand: int = 10) -> Optional[List[str]]:
    """Tiny bounded BFS burst: current -> mid -> target."""
    frontier = []
    for l in _get_links_cached(current_url)[: min(220, len(_get_links_cached(current_url)))]:
        if l["url"] not in visited:
            frontier.append(l["url"])
    for mid in frontier[:budget_expand]:
        try:
            links2 = _get_links_cached(mid)
        except Exception:
            continue
        if any(x["url"] == target_url for x in links2):
            return [current_url, mid, target_url]
    return None


def _solve_one(pair: dict, pair_deadline: float, hard_deadline: float) -> dict:
    pair_id = pair["pair_id"]
    start_url = pair["start"]
    target_url = pair["target"]
    target_title = _url_title(target_url)

    path = [start_url]
    visited = {start_url}
    llm_calls = 0
    link_counts: List[int] = []
    current = start_url

    max_steps = 14  # >=15 yields 0 points anyway

    for step in range(max_steps):
        now = time.time()
        if now >= pair_deadline or now >= hard_deadline - 0.3:
            break

        if current == target_url or _url_title(current).lower() == target_title.lower():
            return _result(pair_id, path, llm_calls, link_counts, True)

        try:
            links = _get_links_cached(current)
        except Exception:
            break

        link_counts.append(len(links))
        if not links:
            break

        # Direct hit?
        direct = next((l for l in links if l["url"] == target_url or _url_title(l["url"]).lower() == target_title.lower()), None)
        if direct:
            path.append(direct["url"])
            return _result(pair_id, path, llm_calls, link_counts, True)

        # Tiny 2-hop burst occasionally, especially early
        if step <= 3:
            burst = _two_hop_best(current, target_url, visited, budget_expand=8)
            if burst is not None:
                # burst starts with current, append rest
                path.extend(burst[1:])
                return _result(pair_id, path, llm_calls, link_counts, True)

        candidates = _rank_candidates(links, target_title, visited, k=14)
        if not candidates:
            # fall back to an unvisited random link
            fallback = next((l for l in links if l["url"] not in visited), random.choice(links))
            current = fallback["url"]
            path.append(current)
            visited.add(current)
            continue

        # Use LLM on a subset, but cap calls to stay within global budget
        use_llm = step < 8 and len(candidates) >= 4
        picked = None
        if use_llm:
            try:
                idx = _llm_pick(_url_title(current), target_title, candidates[:10])
                llm_calls += 1
                if idx is not None:
                    picked = candidates[:10][idx]
            except Exception:
                picked = None

        if picked is None:
            picked = candidates[0]

        current = picked["url"]
        path.append(current)
        visited.add(current)

    success = _url_title(path[-1]).lower() == target_title.lower()
    if len(path) < 2:
        path = [start_url, start_url]
    return _result(pair_id, path, llm_calls, link_counts, success)


# ── entry point ───────────────────────────────────────────────────────────────

def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within the shared deadline.

    Strategy:
    - Prioritize easy/medium pairs first for guaranteed points.
    - Assign each pair a soft local budget based on remaining time/pairs.
    - Always return one result per pair in the original order.
    """
    if not pairs:
        return []

    # Work order for higher expected value first, then map back to original order.
    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    indexed = list(enumerate(pairs))
    ordered = sorted(indexed, key=lambda x: (difficulty_rank.get(x[1].get("difficulty", "hard"), 3), x[0]))

    out: Dict[int, dict] = {}

    for j, (orig_idx, pair) in enumerate(ordered):
        now = time.time()
        if now >= deadline - 0.5:
            out[orig_idx] = _empty_result(pair)
            continue

        remaining = len(ordered) - j
        # leave small reserve buffer
        rem_time = max(0.0, deadline - now - 0.4)
        per_pair = max(2.0, rem_time / max(1, remaining))
        # do not over-invest on single pair
        pair_budget = min(22.0, per_pair * 1.35)
        pair_deadline = min(deadline - 0.35, now + pair_budget)

        try:
            out[orig_idx] = _solve_one(pair, pair_deadline, deadline)
        except Exception:
            out[orig_idx] = _empty_result(pair)

    return [out[i] for i in range(len(pairs))]
