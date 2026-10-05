"""Google Gemini provider (free tier available).

Self-adjusting behavior:
- Discovers models via models.list (a hint, not a guarantee).
- Tries a preference-ordered list of candidates.
- Retries transient 5xx errors with exponential backoff.
- Falls back to the next candidate on 404 (model unavailable).
- Caches the winning model so subsequent messages reuse it.
"""

import asyncio
import json
from typing import AsyncIterator

import httpx

from ultron.providers.base import ChatRequest, LLMProvider
from ultron.utils.logging import setup_logging

logger = setup_logging()

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Models tried in order. The first one that responds successfully wins.
# "gemini-flash-latest" is a stable alias Google maintains — it always
# points to the current best Flash model.
_PREFERRED_MODELS = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

# Status codes worth retrying on the same model.
_TRANSIENT_STATUSES = {500, 502, 503, 504}

# How many times to retry a transient failure on the same model.
_MAX_TRANSIENT_RETRIES = 3

# Base delay between retries in seconds (doubles each attempt).
_RETRY_BASE_DELAY = 1.5


class _ModelUnavailable(Exception):
    """Raised when a model returns 404 — try the next candidate."""


class _TransientError(Exception):
    """Raised when a model returns 5xx — retry the same model."""


class GeminiProvider(LLMProvider):
    """Google Gemini via the Generative Language API."""

    name = "Google Gemini"

    def __init__(self) -> None:
        self._resolved_model: str | None = None

    # ------------------------------------------------------------------
    # Model discovery (kept as a helper, but not trusted blindly)
    # ------------------------------------------------------------------

    async def _fetch_available_models(self, api_key: str) -> list[str]:
        """Ask Google which models this API key can list."""
        url = f"{_GEMINI_BASE}/models?key={api_key}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    logger.warning(
                        "Could not list Gemini models (HTTP %s).",
                        response.status_code,
                    )
                    return []
                data = response.json()
                names: list[str] = []
                for entry in data.get("models", []):
                    raw_name = entry.get("name", "")
                    if raw_name.startswith("models/"):
                        raw_name = raw_name[len("models/"):]
                    actions = entry.get("supportedGenerationMethods", [])
                    if "generateContent" in actions:
                        names.append(raw_name)
                logger.info("Gemini reported %d usable models.", len(names))
                return names
        except Exception as exc:
            logger.warning("Model discovery failed: %s", type(exc).__name__)
            return []

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_contents(self, request: ChatRequest) -> tuple[str, list[dict]]:
        system_text = ""
        contents = []
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
        return system_text, contents

    # ------------------------------------------------------------------
    # Streaming with retry and fallback
    # ------------------------------------------------------------------

    async def stream_chat(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[str]:
        """Stream a chat response, trying multiple models if needed."""
        candidates = self._candidate_list()

        last_error: str | None = None
        for model_name in candidates:
            try:
                logger.info("Trying Gemini model '%s'…", model_name)
                async for chunk in self._stream_with_model(
                    request, api_key, model_name
                ):
                    yield chunk
                # Success — remember it for next time.
                self._resolved_model = model_name
                logger.info("Gemini model '%s' succeeded.", model_name)
                return
            except _ModelUnavailable as exc:
                logger.warning(
                    "Model '%s' unavailable (404). Trying next candidate.",
                    model_name,
                )
                last_error = str(exc)
                continue
            except _TransientError as exc:
                logger.warning(
                    "Model '%s' failed after retries: %s", model_name, exc
                )
                last_error = str(exc)
                continue
            except RuntimeError:
                # Non-recoverable (auth, malformed request). Do not keep trying.
                raise

        raise RuntimeError(
            "No working Gemini model found. All candidates failed. "
            f"Last error: {last_error}"
        )

    def _candidate_list(self) -> list[str]:
        """Return the models to try, resolved model first if known."""
        if self._resolved_model:
            others = [
                m for m in _PREFERRED_MODELS if m != self._resolved_model
            ]
            return [self._resolved_model] + others
        return list(_PREFERRED_MODELS)

    async def _stream_with_model(
        self, request: ChatRequest, api_key: str, model_name: str
    ) -> AsyncIterator[str]:
        """Stream using one model, retrying transient 5xx failures."""
        system_text, contents = self._build_contents(request)

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = (
            f"{_GEMINI_BASE}/models/{model_name}:streamGenerateContent"
            f"?alt=sse&key={api_key}"
        )
        headers = {"Content-Type": "application/json"}

        last_transient: Exception | None = None

        for attempt in range(1, _MAX_TRANSIENT_RETRIES + 1):
            try:
                async for chunk in self._do_stream(url, payload, headers):
                    yield chunk
                return  # success — exit the retry loop
            except _ModelUnavailable:
                raise  # 404 — not retryable, let caller try next model
            except _TransientError as exc:
                last_transient = exc
                if attempt < _MAX_TRANSIENT_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Transient error on '%s' (attempt %d/%d). "
                        "Retrying in %.1fs…",
                        model_name,
                        attempt,
                        _MAX_TRANSIENT_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "Model '%s' still failing after %d attempts.",
                        model_name,
                        _MAX_TRANSIENT_RETRIES,
                    )

        if last_transient:
            raise last_transient

    async def _do_stream(
        self, url: str, payload: dict, headers: dict
    ) -> AsyncIterator[str]:
        """One attempt at streaming. Raises typed errors on failure."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=120.0)
        ) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                status = response.status_code

                if status == 404:
                    raise _ModelUnavailable(
                        f"HTTP 404 — model not found"
                    )

                if status in _TRANSIENT_STATUSES:
                    await response.aread()
                    raise _TransientError(
                        f"HTTP {status} — server temporarily unavailable"
                    )

                if status == 429:
                    await response.aread()
                    raise RuntimeError(
                        "Gemini rate limit reached (HTTP 429). "
                        "Wait a minute and try again."
                    )

                if status == 400:
                    body = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    if "api key" in body.lower():
                        raise RuntimeError(
                            "Gemini rejected the API key (HTTP 400). "
                            "Re-enter it in Settings."
                        )
                    raise RuntimeError(
                        f"Gemini rejected the request (HTTP 400): "
                        f"{body[:200]}"
                    )

                if status == 403:
                    await response.aread()
                    raise RuntimeError(
                        "Gemini denied access (HTTP 403). "
                        "Your key may not have permission for this model."
                    )

                if status != 200:
                    body = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(
                        f"Gemini returned HTTP {status}: {body[:200]}"
                    )

                # 200 — parse the streaming body.
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    candidates = chunk.get("candidates", [])
                    if not candidates:
                        continue
                    parts = (
                        candidates[0].get("content", {}).get("parts", [])
                    )
                    for part in parts:
                        text = part.get("text")
                        if text:
                            yield text

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    async def validate_key(self, api_key: str) -> bool:
        url = f"{_GEMINI_BASE}/models?key={api_key}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False