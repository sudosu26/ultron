"""Microphone recording via sounddevice.

Records 16 kHz mono 16-bit PCM to a temp WAV file (Whisper-friendly).
"""

import tempfile
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

from ultron.utils.logging import setup_logging

logger = setup_logging()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


class Recorder:
    """A simple push-to-talk style recorder."""

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            return
        self._frames = []
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True
            logger.info("Recording started.")
        except Exception as exc:
            logger.error("Failed to start recording: %s", type(exc).__name__)
            raise RuntimeError(
                f"Could not access the microphone: {exc}"
            ) from exc

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Recorder status: %s", status)
        if self._recording:
            self._frames.append(indata.copy())

    def stop(self) -> Optional[str]:
        """Stop recording; return path to WAV file, or None if empty."""
        if not self._recording:
            return None
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

        if not self._frames:
            logger.info("No audio captured.")
            return None

        audio = np.concatenate(self._frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="ultron_"
        )
        tmp.close()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        duration = len(audio) / SAMPLE_RATE
        logger.info("Recording stopped (%.1fs).", duration)
        return tmp.name

    def cancel(self) -> None:
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._frames = []