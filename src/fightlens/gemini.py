import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

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


def generate_text(prompt: str) -> str:
    """Send a text prompt to Gemini and return its response."""

    interaction = get_client().interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return interaction.output_text


def describe_images(image_paths: Sequence[str | Path], prompt: str) -> str:
    """
    Send several images plus a text prompt in ONE multimodal request.

    Images are passed in the given order (chronological for a time window),
    followed by the prompt text, and Gemini's plain-text answer is returned.
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

    interaction = get_client().interactions.create(
        model=GEMINI_MODEL,
        input=content,
    )

    return interaction.output_text
