from google import genai

from fightlens.config import GEMINI_API_KEY, GEMINI_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def generate_text(prompt: str) -> str:
    """Send a text prompt to Gemini and return its response."""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return interaction.output_text