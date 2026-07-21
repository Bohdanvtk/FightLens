"""Local text embeddings for window descriptions and their persistence.

Single home for "text <-> vector" and the vector store I/O, reused by both
Step 4 (embed) and Step 5 (search). Purely local (sentence-transformers) —
never calls Gemini or any paid API.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fightlens.atomic import atomic_write


# Output dimension of the default model (all-MiniLM-L6-v2). Fixed by the
# model, so it's not a config key; model_name lets readers detect a mismatch.
EMBEDDING_DIM = 384


def hash_description(description: str) -> str:
    """SHA-1 hex of a description string — used as its cache key."""

    return hashlib.sha1(description.encode("utf-8")).hexdigest()


class Embedder:
    """Turns descriptions into L2-normalized vectors with a local model.

    Cheap to construct: the SentenceTransformer loads lazily on the first
    encode() call and is cached on the instance.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        """Load and cache the SentenceTransformer on first use."""

        if self._model is None:
            # Imported here so tests using a stub encoder never pull in torch.
            from sentence_transformers import SentenceTransformer

            # "auto" isn't a valid device string for sentence-transformers;
            # None lets it auto-detect cuda/cpu instead.
            device = None if self.device == "auto" else self.device
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into float32 vectors, shape (len(texts), 384), normalized per config."""

        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class EmbeddingStore:
    """One video's window vectors. The four arrays are aligned by row (row i = one window)."""

    vectors: np.ndarray  # float32, shape (N, 384)
    window_ids: np.ndarray  # str, shape (N,)
    desc_hashes: np.ndarray  # str, shape (N,) — sha1 of each stripped desc
    model_name: str

    def __len__(self) -> int:
        return len(self.window_ids)


def save_store(store: EmbeddingStore, path: str | Path) -> None:
    """Write the store to one .npz (vectors, window_ids, desc_hashes, model_name), atomically."""

    atomic_write(
        path,
        lambda file: np.savez(
            file,
            vectors=store.vectors.astype(np.float32, copy=False),
            window_ids=np.asarray(store.window_ids, dtype="U"),
            desc_hashes=np.asarray(store.desc_hashes, dtype="U"),
            model_name=np.asarray(store.model_name),
        ),
        mode="wb",
    )


def load_store(path: str | Path) -> EmbeddingStore:
    """Read a .npz written by save_store back into an EmbeddingStore."""

    with np.load(Path(path), allow_pickle=False) as data:
        return EmbeddingStore(
            vectors=data["vectors"].astype(np.float32, copy=False),
            window_ids=data["window_ids"].astype(str),
            desc_hashes=data["desc_hashes"].astype(str),
            model_name=str(data["model_name"]),
        )
