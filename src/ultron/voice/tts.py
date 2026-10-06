"""Text-to-speech via Microsoft Edge TTS (free) + soundfile playback.

Edge TTS produces MP3. soundfile decodes MP3 (via bundled libsndfile),
and sounddevice plays the decoded PCM.
"""

import tempfile
from pathlib import Path

import aiohttp
import edge_tts
import sounddevice as sd
import soundfile as sf

from ultron.utils.logging import setup_logging

logger = setup_logging()

DEFAULT_VOICE = "en-US-GuyNeural"

# A curated list of good-quality voices.
AVAILABLE_VOICES = [
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-AU-WilliamNeural",
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",
]


async def synthesize_to_file(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Synthesize text to an MP3 file. Returns the temp file path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp3", delete=False, prefix="ultron_tts_"
    )
    tmp.close()
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp.name)
    except aiohttp.WSServerHandshakeError as exc:
        Path(tmp.name).unlink(missing_ok=True)
        if exc.status == 403:
            raise RuntimeError(
                "Microsoft Edge TTS rejected the connection (HTTP 403). "
                "Check for an edge-tts update or try another network."
            ) from exc
        raise RuntimeError(
            f"Microsoft Edge TTS connection failed (HTTP {exc.status})."
        ) from exc
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    logger.info("TTS synthesized (%d chars, %s).", len(text), voice)
    return tmp.name


def play_file_sync(path: str) -> None:
    """Blocking playback of an audio file via soundfile + sounddevice."""
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def stop_playback() -> None:
    """Immediately stop any playback started via sounddevice."""
    try:
        sd.stop()
    except Exception:
        pass