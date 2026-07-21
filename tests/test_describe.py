"""Tests for the Gemini window-description step."""

import json
from pathlib import Path

import pytest

from fightlens import describe
from fightlens.config import validate_descriptions_config, validate_error_log_dir
from fightlens.describe import describe_windows as _describe_windows
from fightlens.describe import window_name


TEST_PROMPT = "Describe the clip."


def describe_windows(manifest_path, output_path, request_delay_seconds):
    """Call the real function with the test prompt and default retries."""

    return _describe_windows(
        manifest_path, output_path, request_delay_seconds, TEST_PROMPT
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manifest(tmp_path: Path, window_count: int = 2) -> Path:
    """Write a minimal manifest plus fake frame images on disk."""

    windows = []
    for window_id in range(window_count):
        folder = tmp_path / "windows" / window_name(window_id)
        folder.mkdir(parents=True)
        image_paths = []
        for position in range(2):
            frame = folder / f"img_{position:02d}_frame_{position:08d}_0.00s.jpg"
            frame.write_bytes(b"fake-jpeg")
            image_paths.append(str(frame))
        windows.append(
            {
                "window_id": window_id,
                "start_timestamp": window_id * 2.0,
                "end_timestamp": (window_id + 1) * 2.0,
                "image_paths": image_paths,
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"windows": windows}), encoding="utf-8"
    )
    return manifest_path


class FakeGemini:
    """Stand-in for gemini.describe_images that records its calls."""

    def __init__(self, fail_first_n_calls: int = 0, error: Exception | None = None):
        self.calls: list[list] = []
        self.fail_first_n_calls = fail_first_n_calls
        self.error = error or RuntimeError("simulated API failure")
        self.timeouts: list = []

    def __call__(self, image_paths, prompt, timeout_seconds=None) -> str:
        self.calls.append(list(image_paths))
        self.timeouts.append(timeout_seconds)
        if len(self.calls) <= self.fail_first_n_calls:
            raise self.error
        return f"description {len(self.calls)}"


@pytest.fixture
def fake_gemini(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    return fake


# ---------------------------------------------------------------------------
# describe_windows
# ---------------------------------------------------------------------------


def test_describe_windows_writes_one_entry_per_window(tmp_path, fake_gemini):
    manifest_path = make_manifest(tmp_path, window_count=2)
    output_path = tmp_path / "descriptions.json"

    summary = describe_windows(manifest_path, output_path, 0)

    assert summary["described"] == 2
    assert summary["failed"] == []
    assert len(fake_gemini.calls) == 2

    entries = json.loads(output_path.read_text(encoding="utf-8"))
    assert [e["window_id"] for e in entries] == ["window_000000", "window_000001"]
    first = entries[0]
    assert first["start_sec"] == 0.0
    assert first["end_sec"] == 2.0
    assert len(first["frames"]) == 2
    assert first["description"]


def test_describe_windows_sends_frames_in_chronological_order(
    tmp_path, fake_gemini
):
    manifest_path = make_manifest(tmp_path, window_count=1)

    describe_windows(manifest_path, tmp_path / "descriptions.json", 0)

    sent = [Path(p).name for p in fake_gemini.calls[0]]
    assert sent == sorted(sent)


def test_describe_windows_is_idempotent(tmp_path, fake_gemini):
    manifest_path = make_manifest(tmp_path, window_count=2)
    output_path = tmp_path / "descriptions.json"

    describe_windows(manifest_path, output_path, 0)
    summary = describe_windows(manifest_path, output_path, 0)

    # Second run makes no API calls at all.
    assert summary["described"] == 0
    assert summary["skipped"] == 2
    assert len(fake_gemini.calls) == 2

    entries = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(entries) == 2


def test_describe_windows_retries_a_failed_call_once(tmp_path, monkeypatch):
    fake = FakeGemini(fail_first_n_calls=1)
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    manifest_path = make_manifest(tmp_path, window_count=1)
    output_path = tmp_path / "descriptions.json"

    summary = describe_windows(manifest_path, output_path, 0)

    # First call fails, the retry succeeds.
    assert len(fake.calls) == 2
    assert summary["described"] == 1
    assert summary["failed"] == []


def test_describe_windows_failed_window_is_skipped_not_fatal(
    tmp_path, monkeypatch
):
    # Window 0 fails twice (call + retry); window 1 succeeds.
    fake = FakeGemini(fail_first_n_calls=2)
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    manifest_path = make_manifest(tmp_path, window_count=2)
    output_path = tmp_path / "descriptions.json"

    summary = describe_windows(manifest_path, output_path, 0)

    assert summary["described"] == 1
    assert summary["failed"] == ["window_000000"]

    # The failed window is retried on the next run, the good one is kept.
    fake.fail_first_n_calls = 0
    summary = describe_windows(manifest_path, output_path, 0)
    assert summary["described"] == 1
    assert summary["failed"] == []
    entries = json.loads(output_path.read_text(encoding="utf-8"))
    assert {e["window_id"] for e in entries} == {"window_000000", "window_000001"}


def test_describe_windows_timeout_is_retried_like_a_failure(
    tmp_path, monkeypatch
):
    # The first call times out, the retry succeeds — the run keeps moving.
    fake = FakeGemini(
        fail_first_n_calls=1,
        error=describe.gemini.GeminiTimeoutError("No Gemini response within 30 s."),
    )
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    manifest_path = make_manifest(tmp_path, window_count=1)

    summary = describe_windows(manifest_path, tmp_path / "out.json", 0)

    assert len(fake.calls) == 2
    assert summary["described"] == 1
    assert summary["failed"] == []


def test_describe_windows_passes_timeout_to_gemini(tmp_path, monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    manifest_path = make_manifest(tmp_path, window_count=1)

    _describe_windows(
        manifest_path,
        tmp_path / "out.json",
        0,
        TEST_PROMPT,
        timeout_seconds=12.5,
    )

    assert fake.timeouts == [12.5]


def test_describe_windows_all_attempts_spent_moves_to_next_window(
    tmp_path, monkeypatch
):
    # Window 0 times out on every attempt; the loop must still reach and
    # describe window 1 instead of waiting on window 0 forever.
    fake = FakeGemini(
        fail_first_n_calls=2,
        error=describe.gemini.GeminiTimeoutError("No Gemini response within 30 s."),
    )
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    manifest_path = make_manifest(tmp_path, window_count=2)

    summary = describe_windows(manifest_path, tmp_path / "out.json", 0)

    assert summary["failed"] == ["window_000000"]
    assert summary["described"] == 1
    assert len(fake.calls) == 3  # 2 spent attempts + 1 for the next window


def test_describe_windows_records_errors_in_error_log(tmp_path, monkeypatch):
    from fightlens.errorlog import ErrorLog

    fake = FakeGemini(fail_first_n_calls=2)
    monkeypatch.setattr(describe.gemini, "describe_images", fake)
    manifest_path = make_manifest(tmp_path, window_count=1)
    error_log = ErrorLog(log_dir=tmp_path / "logs")

    _describe_windows(
        manifest_path,
        tmp_path / "out.json",
        0,
        TEST_PROMPT,
        error_log=error_log,
    )

    assert error_log.count == 2
    entries = json.loads(error_log.path.read_text(encoding="utf-8"))
    assert len(entries) == 2
    first = entries[0]
    assert first["where"] == "window_000000"
    assert first["error_type"] == "RuntimeError"
    assert first["attempt"] == 1
    assert first["timed_out"] is False


def test_describe_windows_skips_windows_without_frames(tmp_path, fake_gemini):
    manifest_path = make_manifest(tmp_path, window_count=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["windows"].append(
        {
            "window_id": 1,
            "start_timestamp": 2.0,
            "end_timestamp": 4.0,
            "image_paths": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = describe_windows(manifest_path, tmp_path / "out.json", 0)

    assert summary["described"] == 1
    assert summary["skipped"] == 1
    assert len(fake_gemini.calls) == 1


def test_describe_windows_missing_manifest_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract"):
        describe_windows(tmp_path / "nope.json", tmp_path / "out.json", 0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _valid_config() -> dict:
    return {
        "manifest_path": "data/processed/test_video/manifest.json",
        "request_delay_seconds": 4.0,
        "retry_attempts": 1,
        "prompt": TEST_PROMPT,
    }


def test_validate_descriptions_config_accepts_valid_config():
    params = validate_descriptions_config(_valid_config())
    assert params["request_delay_seconds"] == 4.0
    assert params["retry_attempts"] == 1
    assert params["prompt"] == TEST_PROMPT


def test_validate_descriptions_config_defaults_delay_and_retries():
    cfg = _valid_config()
    cfg.pop("request_delay_seconds")
    cfg.pop("retry_attempts")
    params = validate_descriptions_config(cfg)
    assert params["request_delay_seconds"] == 4.0
    assert params["retry_attempts"] == 1


def test_validate_descriptions_config_defaults_timeout_to_30():
    params = validate_descriptions_config(_valid_config())
    assert params["response_timeout_seconds"] == 30.0


def test_validate_descriptions_config_allows_null_timeout():
    cfg = _valid_config()
    cfg["response_timeout_seconds"] = None
    params = validate_descriptions_config(cfg)
    assert params["response_timeout_seconds"] is None


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.pop("manifest_path"), "manifest_path"),
        (lambda c: c.update(request_delay_seconds=-1), "request_delay_seconds"),
        (lambda c: c.update(retry_attempts=-1), "retry_attempts"),
        (lambda c: c.update(retry_attempts=1.5), "retry_attempts"),
        (lambda c: c.update(response_timeout_seconds=0), "response_timeout_seconds"),
        (lambda c: c.update(response_timeout_seconds=-5), "response_timeout_seconds"),
        (lambda c: c.pop("prompt"), "prompt"),
        (lambda c: c.update(prompt="  "), "prompt"),
    ],
)
def test_validate_descriptions_config_rejects_bad_values(mutate, match):
    cfg = _valid_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_descriptions_config(cfg)


def test_validate_descriptions_config_requires_mapping():
    with pytest.raises(ValueError, match="descriptions"):
        validate_descriptions_config(None)


def test_validate_error_log_dir_defaults_to_logs():
    assert validate_error_log_dir(None) == "logs"


def test_validate_error_log_dir_accepts_custom_path():
    assert validate_error_log_dir("data/error_logs") == "data/error_logs"


@pytest.mark.parametrize("bad_value", ["", "   ", 5, True, ["logs"]])
def test_validate_error_log_dir_rejects_bad_values(bad_value):
    with pytest.raises(ValueError, match="error_log_dir"):
        validate_error_log_dir(bad_value)
