"""Step 6: optional LLM reranking of the top-N embedding search candidates.

Retrieve-then-rerank: embeddings (Step 5) cheaply narrow a video down to N
candidates, then ONE Gemini text request reorders just those N by the
actual action described (landed vs blocked/slipped, attacker vs defender,
exact punch type) — nuances plain embeddings miss. Off by default; any
failure here (bad answer, timeout, exception) degrades to the embedding
order instead of crashing the search.
"""

import re
from typing import Callable

from fightlens.search import SearchResult


def build_rerank_prompt(query: str, results: list[SearchResult]) -> str:
    """Build the prompt asking Gemini to reorder the numbered candidates by relevance to `query`."""

    lines = [
        "Rerank these boxing video clips by relevance to the query.",
        f'Query: "{query}"',
        "",
        "Judge by the ACTUAL action described — who attacks, punch type, "
        "target, and result (landed / blocked / slipped / missed) — not by "
        "word overlap with the query.",
        "",
        "Candidates:",
    ]
    for index, result in enumerate(results, start=1):
        description = (result.description or "(no description)").strip()
        lines.append(f"{index}. {description}")

    lines += [
        "",
        f"Output every candidate number from 1 to {len(results)}, each once, "
        "ordered MOST to LEAST relevant, one per line. Numbers only — no "
        "words, no punctuation, no explanation. Example:",
        "3",
        "1",
        "2",
    ]
    return "\n".join(lines)


def _parse_order(raw: str, n: int) -> list[int]:
    """Parse Gemini's answer into a full 0-based permutation of range(n).

    Invented/out-of-range numbers are ignored, repeats keep the first
    mention, and any candidate the model never mentioned is appended at
    the end in its original (embedding) order.
    """

    seen_indices: list[int] = []
    seen_set: set[int] = set()
    for match in re.findall(r"-?\d+", raw or ""):
        index = int(match) - 1  # candidates are numbered from 1 in the prompt
        if 0 <= index < n and index not in seen_set:
            seen_set.add(index)
            seen_indices.append(index)

    for index in range(n):
        if index not in seen_set:
            seen_indices.append(index)

    return seen_indices


def rerank(
    query: str,
    results: list[SearchResult],
    gemini_call: Callable[[str, str], str],
    model: str,
) -> list[SearchResult]:
    """
    Reorder `results` by Gemini's relevance judgment; never loses or crashes.

    Only reorders the given SearchResult objects — never recomputes scores
    or re-runs embeddings; each result's original similarity score is kept
    as-is. Any failure (bad/garbage answer, exception, timeout) falls back
    to returning `results` unchanged (the embedding order).
    """

    if len(results) < 2:
        return results

    prompt = build_rerank_prompt(query, results)

    try:
        raw = gemini_call(prompt, model)
    except Exception:  # noqa: BLE001 - any failure degrades to embedding order
        return results

    order = _parse_order(raw, len(results))
    return [results[index] for index in order]
