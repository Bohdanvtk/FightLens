"""Step 5: semantic search over a video's embedded windows.

Purely local and read-only: it loads embeddings.npz and descriptions.json,
ranks windows against a text query with the Step 4 Embedder, and prints the
results. Never calls Gemini or any paid API, writes nothing, touches no
manifest.
"""

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fightlens.embed import load_descriptions
from fightlens.embeddings import Embedder, EmbeddingStore, load_store


@dataclass
class SearchResult:
    """One ranked window: its similarity score plus fields joined from descriptions.json."""

    window_id: str
    score: float
    start_sec: float | None
    end_sec: float | None
    description: str | None


class Retriever:
    """Ranks a video's embedded windows against a text query using the Step 4 Embedder."""

    def __init__(
        self,
        store: EmbeddingStore,
        descriptions: list[dict[str, Any]],
        embedder: Embedder,
    ):
        self.store = store
        self.embedder = embedder
        self._by_id = {entry["window_id"]: entry for entry in descriptions}

        # Non-fatal staleness check: descriptions regenerated without a
        # matching re-embed would silently drop or misjoin windows below.
        store_ids = set(str(window_id) for window_id in store.window_ids)
        description_ids = set(self._by_id)
        if store_ids != description_ids:
            print(
                "Warning: embeddings.npz and descriptions.json cover "
                "different windows — run `python -m fightlens embed` to "
                "refresh the index.",
                file=sys.stderr,
            )

    @classmethod
    def from_paths(
        cls,
        embeddings_path: str | Path,
        descriptions_path: str | Path,
        embedder: Embedder,
    ) -> "Retriever":
        """Load the store and descriptions.json from disk and build a Retriever."""

        embeddings_path = Path(embeddings_path)
        if not embeddings_path.is_file():
            raise FileNotFoundError(
                f"Embeddings not found: {embeddings_path}. "
                "Generate them first (python -m fightlens embed)."
            )

        store = load_store(embeddings_path)
        descriptions = load_descriptions(descriptions_path)
        return cls(store, descriptions, embedder)

    def __len__(self) -> int:
        return len(self.store)

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Encode `query` and return the top `k` windows by descending cosine similarity."""

        if len(self.store) == 0:
            return []

        query_vec = self.embedder.encode([query])[0]
        scores = self.store.vectors @ query_vec

        k = min(k, len(scores))
        top_indices = np.argsort(-scores)[:k]

        results = []
        for index in top_indices:
            window_id = str(self.store.window_ids[index])
            entry = self._by_id.get(window_id, {})
            results.append(
                SearchResult(
                    window_id=window_id,
                    score=float(scores[index]),
                    start_sec=entry.get("start_sec"),
                    end_sec=entry.get("end_sec"),
                    description=entry.get("description"),
                )
            )
        return results


def _format_timecode(seconds: float) -> str:
    """Format seconds as MM:SS.s, e.g. 9.6 -> "00:09.6", 74.6 -> "01:14.6"."""

    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"


def format_results(
    query: str,
    results: list[SearchResult],
    total: int,
    reranked: bool = False,
) -> None:
    """Print the ranked results; the only function in this module that prints."""

    print(f'Query: "{query}"')
    suffix = " (reranked)" if reranked else ""
    print(f"Top {len(results)} of {total} windows{suffix}")
    print()

    if not results:
        print("No windows to search — run `python -m fightlens embed` first.")
        return

    for rank, result in enumerate(results, start=1):
        if result.start_sec is not None and result.end_sec is not None:
            timecode = (
                f"{_format_timecode(result.start_sec)}"
                f"–{_format_timecode(result.end_sec)}"
            )
        else:
            timecode = "--:--.-–--:--.-"

        print(f" #{rank}   {result.score:.2f}   {timecode}   {result.window_id}")
        print(
            textwrap.fill(
                result.description or "(no description)",
                width=80,
                initial_indent="      ",
                subsequent_indent="      ",
            )
        )
        print()
