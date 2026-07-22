"""Tests for the optional LLM reranker (Step 6).

A fake gemini_call (a plain callable returning a canned string) stands in
for fightlens.gemini.generate_text everywhere, so no real API is ever hit.
"""

import pytest

from fightlens.config import validate_rerank_config
from fightlens.rerank import rerank
from fightlens.search import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(window_id: str, score: float) -> SearchResult:
    return SearchResult(
        window_id=window_id,
        score=score,
        start_sec=0.0,
        end_sec=1.2,
        description=f"description for {window_id}",
    )


def candidates() -> list[SearchResult]:
    # In embedding (original) order: 0, 1, 2.
    return [
        make_result("window_000000", 0.9),
        make_result("window_000001", 0.8),
        make_result("window_000002", 0.7),
    ]


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------


def test_wellformed_answer_reorders_candidates():
    results = candidates()

    reranked = rerank("query", results, lambda prompt, model: "3\n1\n2", model="m")

    assert [r.window_id for r in reranked] == [
        "window_000002",
        "window_000000",
        "window_000001",
    ]


def test_answer_omitting_candidates_appends_them_in_original_order():
    results = candidates()

    # The model only mentions candidate 2 (window_000001).
    reranked = rerank("query", results, lambda prompt, model: "2", model="m")

    assert [r.window_id for r in reranked] == [
        "window_000001",  # explicitly ranked first
        "window_000000",  # missing -> appended, original order
        "window_000002",  # missing -> appended, original order
    ]


def test_answer_with_invalid_ids_ignores_them_and_returns_full_permutation():
    results = candidates()

    # 5 and 0 are out of range for 3 candidates (valid ids are 1..3); 1
    # repeats and must only count once (first mention wins).
    reranked = rerank(
        "query", results, lambda prompt, model: "5\n2\n0\n1\n3\n1", model="m"
    )

    ids = [r.window_id for r in reranked]
    assert ids == ["window_000001", "window_000000", "window_000002"]
    # Still a full permutation: nothing lost, nothing duplicated.
    assert sorted(ids) == sorted(r.window_id for r in results)


def test_garbage_answer_falls_back_to_original_order():
    results = candidates()

    reranked = rerank("query", results, lambda prompt, model: "not a number here", model="m")

    assert [r.window_id for r in reranked] == [r.window_id for r in results]


def test_empty_answer_falls_back_to_original_order():
    results = candidates()

    reranked = rerank("query", results, lambda prompt, model: "", model="m")

    assert [r.window_id for r in reranked] == [r.window_id for r in results]


def test_gemini_call_raising_falls_back_to_original_order():
    results = candidates()

    def failing_call(prompt, model):
        raise RuntimeError("simulated Gemini failure")

    reranked = rerank("query", results, failing_call, model="m")

    assert [r.window_id for r in reranked] == [r.window_id for r in results]


def test_scores_are_preserved_across_reordering():
    results = candidates()
    original_scores = {r.window_id: r.score for r in results}

    reranked = rerank("query", results, lambda prompt, model: "3\n1\n2", model="m")

    for result in reranked:
        assert result.score == original_scores[result.window_id]


def test_single_candidate_skips_the_api_call():
    results = [make_result("window_000000", 0.9)]

    def must_not_be_called(prompt, model):
        raise AssertionError("gemini_call must not be invoked for < 2 candidates")

    reranked = rerank("query", results, must_not_be_called, model="m")

    assert reranked == results


def test_empty_candidates_skips_the_api_call():
    def must_not_be_called(prompt, model):
        raise AssertionError("gemini_call must not be invoked for 0 candidates")

    assert rerank("query", [], must_not_be_called, model="m") == []


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _valid_config() -> dict:
    return {"enabled": True, "top_n": 10}


def test_validate_rerank_config_accepts_valid_config():
    params = validate_rerank_config(_valid_config())
    assert params == {"enabled": True, "top_n": 10}


def test_validate_rerank_config_applies_defaults():
    params = validate_rerank_config({})
    assert params == {"enabled": False, "top_n": 10}


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.update(enabled="yes"), "enabled"),
        (lambda c: c.update(enabled=1), "enabled"),
        (lambda c: c.update(top_n=0), "top_n"),
        (lambda c: c.update(top_n=-1), "top_n"),
        (lambda c: c.update(top_n=1.5), "top_n"),
        (lambda c: c.update(top_n=True), "top_n"),
    ],
)
def test_validate_rerank_config_rejects_bad_values(mutate, match):
    cfg = _valid_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_rerank_config(cfg)


def test_validate_rerank_config_requires_mapping():
    with pytest.raises(ValueError, match="rerank"):
        validate_rerank_config(None)
