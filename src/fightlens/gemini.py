import base64
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Mime types for the image formats the extraction step can produce.
_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


_T = TypeVar("_T")


class GeminiTimeoutError(RuntimeError):
    """Raised when Gemini does not answer within the allowed time."""


def _call_with_timeout(func: Callable[[], _T], timeout_seconds: float) -> _T:
    """
    Run one blocking Gemini call, but give up after timeout_seconds.

    The call runs in a daemon thread while this function watches the
    clock. If the response does not arrive in time, the stuck request is
    abandoned (an HTTP call already in flight cannot be cancelled) and
    GeminiTimeoutError is raised so the caller can immediately send a
    fresh request instead of stalling the whole run.
    """

    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = func()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            result["error"] = error

    thread = threading.Thread(
        target=target, daemon=True, name="gemini-request"
    )
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise GeminiTimeoutError(
            f"No Gemini response within {timeout_seconds:g} s."
        )
    if "error" in result:
        raise result["error"]
    return result["value"]


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """
    Build the Gemini client on first use.

    The API key is only required when Gemini is actually called, so steps
    that never talk to Gemini (e.g. frame extraction) run without it.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the .env file."
        )

    return genai.Client(api_key=api_key)


def describe_images(
    image_paths: Sequence[str | Path],
    prompt: str,
    timeout_seconds: float | None = None,
) -> str:
    """
    Send several images plus a text prompt in ONE multimodal request.

    Images are passed in the given order (chronological for a time window),
    followed by the prompt text, and Gemini's plain-text answer is returned.

    If timeout_seconds is set and no answer arrives in time, the request
    is abandoned and GeminiTimeoutError is raised (null/None = wait
    forever, the old behaviour).
    """

    if not image_paths:
        raise ValueError("describe_images needs at least one image path.")

    content: list[dict] = []
    for image_path in image_paths:
        image_path = Path(image_path)
        mime_type = _IMAGE_MIME_TYPES.get(image_path.suffix.lower())
        if mime_type is None:
            raise ValueError(
                f"Unsupported image format {image_path.suffix!r}: {image_path}"
            )
        content.append(
            {
                "type": "image",
                "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "mime_type": mime_type,
            }
        )

    content.append({"type": "text", "text": prompt})

    def _request() -> str:
        interaction = get_client().interactions.create(
            model=GEMINI_MODEL,
            input=content,
        )
        return interaction.output_text

    if timeout_seconds is None:
        return _request()
    return _call_with_timeout(_request, timeout_seconds)
