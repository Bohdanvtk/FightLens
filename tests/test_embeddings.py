"""Tests for the local window-embedding step (Step 4).

A stub encoder is injected everywhere so no sentence-transformers model is
ever downloaded or loaded: the tests exercise the caching/alignment logic,
not the neural network.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fightlens.config import validate_embedding_config, validate_preview_config
from fightlens.describe import window_name
from fightlens.embed import embed_windows
from fightlens.embeddings import EMBEDDING_DIM, Embedder, load_store


# ---------------------------------------------------------------------------
# Stub encoder + helpers
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Deterministic stand-in for Embedder that never loads a real model.

    Each text maps to a fixed unit vector, so a build is fully reproducible.
    It records every text it is asked to encode, which lets tests assert
    exactly which windows were (re)computed.
    """

    def __init__(self, model_name: str = "stub-model"):
        self.model_name = model_name
        self.encoded: list[str] = []
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.encoded.extend(texts)
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts]).astype(np.float32)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        vector = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM)
        return (vector / np.linalg.norm(vector)).astype(np.float32)


def write_descriptions(path: Path, descriptions: dict[int, str]) -> None:
    """Write a descriptions.json array; keys are window ids, values the text.

    The dict is deliberately iterated in insertion order (which may not be
    sorted) to prove the step sorts windows itself.
    """

    entries = []
    for window_id, description in descriptions.items():
        entries.append(
            {
                "window_id": window_name(window_id),
                "start_sec": window_id * 1.2,
                "end_sec": (window_id + 1) * 1.2,
                "frames": [],
                "description": description,
            }
        )
    path.write_text(json.dumps(entries), encoding="utf-8")


def make_manifest(path: Path) -> None:
    """Write a minimal manifest for the embeddings artifact to register into."""

    path.write_text(json.dumps({"windows": []}), encoding="utf-8")


@pytest.fixture
def paths(tmp_path):
    descriptions_path = tmp_path / "descriptions.json"
    embeddings_path = tmp_path / "embeddings.npz"
    manifest_path = tmp_path / "manifest.json"
    make_manifest(manifest_path)
    return descriptions_path, embeddings_path, manifest_path


def run(paths, embedder, model_name=None):
    descriptions_path, embeddings_path, manifest_path = paths
    return embed_windows(
        descriptions_path=descriptions_path,
        embeddings_path=embeddings_path,
        manifest_path=manifest_path,
        embedder=embedder,
        model_name=model_name or embedder.model_name,
    )


# ---------------------------------------------------------------------------
# embed_windows
# ---------------------------------------------------------------------------


def test_build_aligns_window_ids_and_vectors(paths):
    descriptions_path, embeddings_path, _ = paths
    # Deliberately out of order to prove the output is sorted by window id.
    write_descriptions(
        descriptions_path,
        {2: "clinch near the ropes", 0: "left jab", 1: "right cross to the body"},
    )
    stub = StubEmbedder()

    summary = run(paths, stub)

    assert summary == {
        "total": 3,
        "reused": 0,
        "embedded": 3,
        "dropped": 0,
        "output_path": str(embeddings_path),
    }

    store = load_store(embeddings_path)
    assert list(store.window_ids) == [
        "window_000000",
        "window_000001",
        "window_000002",
    ]
    assert store.vectors.shape == (3, EMBEDDING_DIM)
    assert store.vectors.dtype == np.float32

    # Row i's vector must be the encoding of row i's window description.
    expected = {
        "window_000000": "left jab",
        "window_000001": "right cross to the body",
        "window_000002": "clinch near the ropes",
    }
    for row, window_id in enumerate(store.window_ids):
        np.testing.assert_array_equal(
            store.vectors[row], StubEmbedder._vector(expected[window_id])
        )


def test_second_run_recomputes_nothing(paths):
    descriptions_path, embeddings_path, _ = paths
    write_descriptions(descriptions_path, {0: "left jab", 1: "body shot"})

    run(paths, StubEmbedder())
    first = load_store(embeddings_path)

    stub = StubEmbedder()
    summary = run(paths, stub)

    # Nothing is embedded the second time, and the file is unchanged.
    assert stub.encoded == []
    assert summary["embedded"] == 0
    assert summary["reused"] == 2

    second = load_store(embeddings_path)
    np.testing.assert_array_equal(first.vectors, second.vectors)
    assert list(first.window_ids) == list(second.window_ids)
    assert list(first.desc_hashes) == list(second.desc_hashes)


def test_changing_one_description_reembeds_only_that_window(paths):
    descriptions_path, embeddings_path, _ = paths
    write_descriptions(descriptions_path, {0: "left jab", 1: "body shot", 2: "clinch"})
    run(paths, StubEmbedder())
    before = load_store(embeddings_path)

    # Change only window 1.
    write_descriptions(
        descriptions_path, {0: "left jab", 1: "overhand right", 2: "clinch"}
    )
    stub = StubEmbedder()
    summary = run(paths, stub)

    # Exactly one window is re-embedded — the changed one.
    assert stub.encoded == ["overhand right"]
    assert summary["embedded"] == 1
    assert summary["reused"] == 2

    after = load_store(embeddings_path)
    ids = list(after.window_ids)
    # Untouched windows keep their exact vectors.
    for window_id in ("window_000000", "window_000002"):
        np.testing.assert_array_equal(
            after.vectors[ids.index(window_id)],
            before.vectors[list(before.window_ids).index(window_id)],
        )
    # The changed window now holds the new encoding.
    np.testing.assert_array_equal(
        after.vectors[ids.index("window_000001")],
        StubEmbedder._vector("overhand right"),
    )


def test_changing_model_forces_full_recompute(paths):
    descriptions_path, embeddings_path, _ = paths
    write_descriptions(descriptions_path, {0: "left jab", 1: "body shot"})
    run(paths, StubEmbedder(), model_name="model-a")

    stub = StubEmbedder()
    summary = run(paths, stub, model_name="model-b")

    # A different model invalidates the whole cache.
    assert summary["embedded"] == 2
    assert summary["reused"] == 0
    assert sorted(stub.encoded) == ["body shot", "left jab"]
    assert load_store(embeddings_path).model_name == "model-b"


def test_removing_a_window_drops_it(paths):
    descriptions_path, embeddings_path, _ = paths
    write_descriptions(descriptions_path, {0: "left jab", 1: "body shot", 2: "clinch"})
    run(paths, StubEmbedder())

    write_descriptions(descriptions_path, {0: "left jab", 1: "body shot"})
    summary = run(paths, StubEmbedder())

    assert summary["dropped"] == 1
    assert summary["total"] == 2
    store = load_store(embeddings_path)
    assert list(store.window_ids) == ["window_000000", "window_000001"]


def test_empty_descriptions_does_not_crash(paths):
    descriptions_path, embeddings_path, _ = paths
    write_descriptions(descriptions_path, {})
    stub = StubEmbedder()

    summary = run(paths, stub)

    assert summary == {
        "total": 0,
        "reused": 0,
        "embedded": 0,
        "dropped": 0,
        "output_path": str(embeddings_path),
    }
    assert stub.calls == 0  # the model is never even consulted
    store = load_store(embeddings_path)
    assert len(store) == 0
    assert store.vectors.shape == (0, EMBEDDING_DIM)


def test_embeddings_artifact_is_registered_in_manifest(paths):
    descriptions_path, _, manifest_path = paths
    write_descriptions(descriptions_path, {0: "left jab", 1: "body shot"})

    run(paths, StubEmbedder())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"]["embeddings"]
    assert artifact == {
        "path": "embeddings.npz",
        "model": "stub-model",
        "dim": EMBEDDING_DIM,
        "count": 2,
    }
    # Registration must not disturb the existing manifest fields.
    assert manifest["windows"] == []


def test_missing_descriptions_errors(paths):
    with pytest.raises(FileNotFoundError, match="describe"):
        run(paths, StubEmbedder())


# ---------------------------------------------------------------------------
# Embedder (no real model is loaded)
# ---------------------------------------------------------------------------


def test_embedder_encode_empty_does_not_load_model():
    # A bogus model name would fail if loading were attempted; the empty
    # shortcut must return before touching sentence-transformers.
    embedder = Embedder(model_name="does-not-exist")
    result = embedder.encode([])
    assert result.shape == (0, EMBEDDING_DIM)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _valid_config() -> dict:
    return {
        "model_name": "all-MiniLM-L6-v2",
        "batch_size": 32,
        "device": "auto",
        "normalize": True,
    }


def test_validate_embedding_config_accepts_valid_config():
    params = validate_embedding_config(_valid_config())
    assert params["model_name"] == "all-MiniLM-L6-v2"
    assert params["batch_size"] == 32
    assert params["device"] == "auto"
    assert params["normalize"] is True


def test_validate_embedding_config_applies_defaults():
    params = validate_embedding_config({"model_name": "all-MiniLM-L6-v2"})
    assert params["batch_size"] == 32
    assert params["device"] == "auto"
    assert params["normalize"] is True


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.pop("model_name"), "model_name"),
        (lambda c: c.update(model_name="  "), "model_name"),
        (lambda c: c.update(batch_size=0), "batch_size"),
        (lambda c: c.update(batch_size=1.5), "batch_size"),
        (lambda c: c.update(batch_size=True), "batch_size"),
        (lambda c: c.update(device=""), "device"),
        (lambda c: c.update(device=5), "device"),
        (lambda c: c.update(normalize="yes"), "normalize"),
    ],
)
def test_validate_embedding_config_rejects_bad_values(mutate, match):
    cfg = _valid_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_embedding_config(cfg)


def test_validate_embedding_config_requires_mapping():
    with pytest.raises(ValueError, match="embedding"):
        validate_embedding_config(None)


def test_validate_preview_config_accepts_valid_config():
    params = validate_preview_config({"fps": 3.0, "player": "mp4"})
    assert params == {"fps": 3.0, "player": "mp4"}


def test_validate_preview_config_applies_defaults():
    params = validate_preview_config({})
    assert params == {"fps": 6.0, "player": "gif"}


def test_validate_preview_config_accepts_auto_fps():
    params = validate_preview_config({"fps": "auto"})
    assert params == {"fps": "auto", "player": "gif"}


@pytest.mark.parametrize(
    "value",
    [0, -1, "2", True, None],
)
def test_validate_preview_config_rejects_bad_fps(value):
    with pytest.raises(ValueError, match="fps"):
        validate_preview_config({"fps": value})


@pytest.mark.parametrize(
    "value",
    ["avi", "", 1, True, None],
)
def test_validate_preview_config_rejects_bad_player(value):
    with pytest.raises(ValueError, match="player"):
        validate_preview_config({"player": value})


def test_validate_preview_config_requires_mapping():
    with pytest.raises(ValueError, match="preview"):
        validate_preview_config(None)
