"""Tests for the Gemini call timeout watchdog."""

import time

import pytest

from fightlens.gemini import GeminiTimeoutError, _call_with_timeout


def test_fast_call_returns_its_value():
    assert _call_with_timeout(lambda: "answer", timeout_seconds=5) == "answer"


def test_slow_call_raises_timeout_error():
    def stuck_call() -> str:
        time.sleep(2)
        return "too late"

    started = time.monotonic()
    with pytest.raises(GeminiTimeoutError, match="0.1"):
        _call_with_timeout(stuck_call, timeout_seconds=0.1)
    # The caller got control back right after the timeout, not after 2 s.
    assert time.monotonic() - started < 1


def test_call_error_is_propagated():
    def bad_call() -> str:
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        _call_with_timeout(bad_call, timeout_seconds=5)
