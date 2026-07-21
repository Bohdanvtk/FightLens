"""Step 4: embed window descriptions into a local vector store.

Runs after description generation, as its own step. Purely local (no Gemini
calls) and idempotent: a window is only (re)embedded when its description
text or the configured model changed, otherwise its cached vector is reused.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from fightlens.embeddings import (
    EMBEDDING_DIM,
    Embedder,
    EmbeddingStore,
    hash_description,
    load_store,
    save_store,
)
from fightlens.video import register_artifact


def load_descriptions(descriptions_path: str | Path) -> list[dict[str, Any]]:
    """Load a video's descriptions.json (a JSON array of window objects)."""

    descriptions_path = Path(descriptions_path)
    if not descriptions_path.is_file():
        raise FileNotFoundError(
            f"Descriptions not found: {descriptions_path}. "
            "Generate them first (python -m fightlens describe)."
        )

    with descriptions_path.open("r", encoding="utf-8") as file:
        entries = json.load(file)

    if not isinstance(entries, list):
        raise ValueError(
            f"The descriptions file must contain a JSON list: {descriptions_path}"
        )

    return entries


def embed_windows(
    descriptions_path: str | Path,
    embeddings_path: str | Path,
    manifest_path: str | Path,
    embedder: Embedder,
    model_name: str,
) -> dict[str, Any]:
    """
    Build (or refresh) embeddings.npz from one video's descriptions.json.

    A window is reused from any previous embeddings.npz when its description
    hash is unchanged AND model_name matches (a different model invalidates
    the whole cache); everything else is (re)embedded in one batch. The
    output holds exactly the current window_ids, sorted, with removed windows
    dropped. Returns a summary with "total", "reused", "embedded", "dropped".
    """

    embeddings_path = Path(embeddings_path)

    entries = load_descriptions(descriptions_path)

    # Current windows, as (window_id, stripped description, description hash).
    current: list[tuple[str, str, str]] = []
    for entry in entries:
        window_id = entry["window_id"]
        description = (entry.get("description") or "").strip()
        current.append((window_id, description, hash_description(description)))

    # Cache from a previous run, but only if it used the same model — vectors
    # from a different model are not comparable, so treat it as empty then.
    cache: dict[str, tuple[str, np.ndarray]] = {}
    if embeddings_path.is_file():
        previous = load_store(embeddings_path)
        if previous.model_name == model_name:
            for index in range(len(previous)):
                cache[str(previous.window_ids[index])] = (
                    str(previous.desc_hashes[index]),
                    previous.vectors[index],
                )

    # Reuse a cached vector when the description is unchanged; otherwise embed.
    vectors_by_id: dict[str, np.ndarray] = {}
    to_embed: list[tuple[str, str]] = []
    reused = 0
    for window_id, description, desc_hash in current:
        cached = cache.get(window_id)
        if cached is not None and cached[0] == desc_hash:
            vectors_by_id[window_id] = cached[1]
            reused += 1
        else:
            to_embed.append((window_id, description))

    # One batched encode call for everything that changed or is new.
    if to_embed:
        new_vectors = embedder.encode([description for _, description in to_embed])
        for (window_id, _), vector in zip(to_embed, new_vectors):
            vectors_by_id[window_id] = vector

    # Assemble the final store: exactly the current windows, sorted by id.
    ordered = sorted(current, key=lambda item: item[0])
    window_ids = [window_id for window_id, _, _ in ordered]
    desc_hashes = [desc_hash for _, _, desc_hash in ordered]
    if window_ids:
        vectors = np.stack([vectors_by_id[wid] for wid in window_ids]).astype(
            np.float32, copy=False
        )
    else:
        vectors = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    store = EmbeddingStore(
        vectors=vectors,
        window_ids=np.asarray(window_ids, dtype="U"),
        desc_hashes=np.asarray(desc_hashes, dtype="U"),
        model_name=model_name,
    )
    save_store(store, embeddings_path)

    # Windows that had a cached vector but are gone from the current input.
    current_ids = {window_id for window_id, _, _ in current}
    dropped = sum(1 for window_id in cache if window_id not in current_ids)

    # Final step: advertise the freshly written .npz in the manifest.
    register_artifact(
        manifest_path,
        "embeddings",
        {
            "path": embeddings_path.name,
            "model": model_name,
            "dim": EMBEDDING_DIM,
            "count": len(window_ids),
        },
    )

    return {
        "total": len(current),
        "reused": reused,
        "embedded": len(to_embed),
        "dropped": dropped,
        "output_path": str(embeddings_path),
    }
