"""Speech-to-text via Groq Whisper (free tier).

Uses the OpenAI-compatible /audio/transcriptions endpoint.
"""

from pathlib import Path

import httpx

from ultron.utils.logging import setup_logging

logger = setup_logging()

_GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_DEFAULT_MODEL = "whisper-large-v3-turbo"


async def transcribe(
    wav_path: str, api_key: str, model: str = _DEFAULT_MODEL
) -> str:
    """Transcribe a WAV file. Returns the plain-text transcript."""
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "response_format": "text"}

    with path.open("rb") as f:
        files = {"file": (path.name, f, "audio/wav")}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _GROQ_URL, headers=headers, data=data, files=files
            )

    if response.status_code == 401:
        raise RuntimeError(
            "Groq rejected the API key. Re-enter it in Settings."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "Groq rate limit reached. Wait a minute and try again."
        )
    if response.status_code != 200:
        body = response.text[:200]
        raise RuntimeError(
            f"Groq returned HTTP {response.status_code}: {body}"
        )

    text = response.text.strip()
    logger.info("STT completed (%d chars).", len(text))
    return text