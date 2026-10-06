"""Google Gemini provider (free tier available).

Multi-key behavior:
- Accepts a list of API keys.
- Remembers which key succeeded last and tries it first next time.
- On HTTP 429, moves to the next key immediately (no waiting).
- If all keys are rate-limited, waits once for the shortest requested
  delay, then retries all keys one final time.
- Falls back to the next model on 404.
- Retries transient 5xx errors a limited number of times.

Logs are safe: only masked keys (first 8 + last 4 chars) are logged.
"""

import asyncio
import json
import re
from typing import AsyncIterator

import httpx

from ultron.providers.base import ChatRequest, LLMProvider
from ultron.utils.logging import setup_logging

logger = setup_logging()

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

_PREFERRED_MODELS = [
    # Recent local runs succeeded with this model after the aliases below
    # repeatedly returned 503, so try the known-working candidate first.
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

_TRANSIENT_STATUSES = {500, 502, 503, 504}
_MAX_TRANSIENT_RETRIES = 2
_RETRY_BASE_DELAY = 2.0

_MAX_RATE_LIMIT_WAIT = 60.0


# --------------------------------------------------------------------------
# Internal exceptions
# --------------------------------------------------------------------------

class _ModelUnavailable(Exception):
    """404 — model not found. Try the next model."""


class _TransientError(Exception):
    """5xx after retries — model is broken right now. Try next model."""


class _RateLimited(Exception):
    """429 — key exhausted. Try next key."""

    def __init__(self, wait_seconds: float) -> None:
        super().__init__(f"rate limited; retry after {wait_seconds:.0f}s")
        self.wait_seconds = wait_seconds


class _AllModelsFailed(Exception):
    """Every model failed for this key without a rate limit."""


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    name = "Google Gemini"

    def __init__(self) -> None:
        self._resolved_model: str | None = None
        self._preferred_key_index: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def stream_chat(
        self, request: ChatRequest, api_keys: list[str]
    ) -> AsyncIterator[str]:
        if not api_keys:
            raise RuntimeError(
                "No Gemini API keys configured. Open ⚙ Settings to add one."
            )

        n = len(api_keys)
        start = self._preferred_key_index % n
        shortest_wait: float | None = None

        for round_num in range(2):  # First pass, then one retry after wait.
            for offset in range(n):
                key_index = (start + offset) % n
                api_key = api_keys[key_index]
                masked = _mask_key(api_key)
                logger.info("Trying Gemini key %s…", masked)

                try:
                    async for chunk in self._try_key_all_models(
                        request, api_key
                    ):
                        yield chunk
                    self._preferred_key_index = key_index
                    logger.info("Succeeded with key %s.", masked)
                    return
                except _RateLimited as rl:
                    logger.warning(
                        "Key %s is rate-limited (asked to wait %.0fs).",
                        masked, rl.wait_seconds,
                    )
                    if shortest_wait is None or rl.wait_seconds < shortest_wait:
                        shortest_wait = rl.wait_seconds
                    continue
                except _AllModelsFailed as exc:
                    logger.warning("Key %s: %s", masked, exc)
                    continue

            if round_num == 0 and shortest_wait is not None:
                wait = min(shortest_wait, _MAX_RATE_LIMIT_WAIT)
                logger.warning(
                    "All %d key(s) rate-limited. Waiting %.0fs, then one "
                    "final attempt…", n, wait,
                )
                await asyncio.sleep(wait)
            else:
                break

        raise RuntimeError(
            f"All {n} Gemini key(s) are rate-limited. Free-tier keys allow "
            f"only a few requests per minute. Wait a minute, or add another "
            f"key in ⚙ Settings."
        )

    # ------------------------------------------------------------------
    # Key-level: try every model with one key
    # ------------------------------------------------------------------

    async def _try_key_all_models(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[str]:
        candidates = self._candidate_list()
        last_error: str | None = None

        for model_name in candidates:
            logger.info("Trying model '%s'…", model_name)
            try:
                async for chunk in self._stream_with_model(
                    request, api_key, model_name
                ):
                    yield chunk
                self._resolved_model = model_name
                logger.info("Model '%s' succeeded.", model_name)
                return
            except _RateLimited:
                # Key exhausted; stop trying models with this key.
                raise
            except _ModelUnavailable:
                continue
            except _TransientError as exc:
                last_error = str(exc)
                continue
            except RuntimeError:
                # Auth or malformed request — do not keep trying.
                raise

        raise _AllModelsFailed(
            last_error or "all model candidates unavailable"
        )

    def _candidate_list(self) -> list[str]:
        if self._resolved_model:
            others = [m for m in _PREFERRED_MODELS if m != self._resolved_model]
            return [self._resolved_model] + others
        return list(_PREFERRED_MODELS)

    # ------------------------------------------------------------------
    # Model-level: stream with retries on transient 5xx
    # ------------------------------------------------------------------

    async def _stream_with_model(
        self, request: ChatRequest, api_key: str, model_name: str
    ) -> AsyncIterator[str]:
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

        attempt = 0
        while True:
            try:
                async for chunk in self._do_stream(url, payload, headers):
                    yield chunk
                return
            except (_ModelUnavailable, _RateLimited):
                raise
            except _TransientError:
                attempt += 1
                if attempt >= _MAX_TRANSIENT_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * attempt
                logger.warning(
                    "Transient error on '%s' (attempt %d/%d). "
                    "Retrying in %.1fs…",
                    model_name, attempt, _MAX_TRANSIENT_RETRIES, delay,
                )
                await asyncio.sleep(delay)

    async def _do_stream(
        self, url: str, payload: dict, headers: dict
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=180.0)
        ) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                status = response.status_code

                if status == 404:
                    raise _ModelUnavailable("HTTP 404 — model not found")

                if status in _TRANSIENT_STATUSES:
                    await response.aread()
                    raise _TransientError(
                        f"HTTP {status} — server temporarily unavailable"
                    )

                if status == 429:
                    body = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    wait = _parse_retry_delay(
                        response.headers.get("retry-after"), body
                    )
                    raise _RateLimited(wait)

                if status == 400:
                    body = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    if "api key" in body.lower():
                        raise RuntimeError(
                            "Gemini rejected an API key (HTTP 400). "
                            "Re-enter keys in Settings."
                        )
                    raise RuntimeError(
                        f"Gemini rejected the request (HTTP 400): {body[:200]}"
                    )

                if status == 403:
                    await response.aread()
                    raise RuntimeError(
                        "Gemini denied access (HTTP 403). Your key may not "
                        "have permission for this model."
                    )

                if status != 200:
                    body = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(
                        f"Gemini returned HTTP {status}: {body[:200]}"
                    )

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
    # Helpers
    # ------------------------------------------------------------------

    def _build_contents(
        self, request: ChatRequest
    ) -> tuple[str, list[dict]]:
        system_text = ""
        contents = []
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            elif msg.role == "user":
                contents.append(
                    {"role": "user", "parts": [{"text": msg.content}]}
                )
            elif msg.role == "assistant":
                contents.append(
                    {"role": "model", "parts": [{"text": msg.content}]}
                )
        return system_text, contents

    async def validate_key(self, api_key: str) -> bool:
        url = f"{_GEMINI_BASE}/models?key={api_key}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """Return a safe representation of a key for logging."""
    if not key or len(key) < 12:
        return "***"
    return f"{key[:8]}…{key[-4:]}"


def _parse_retry_delay(retry_after_header: str | None, body: str) -> float:
    if retry_after_header:
        try:
            value = float(retry_after_header.strip())
            return max(1.0, min(value, _MAX_RATE_LIMIT_WAIT))
        except ValueError:
            pass

    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body)
    if match:
        try:
            value = float(match.group(1))
            return max(1.0, min(value, _MAX_RATE_LIMIT_WAIT))
        except ValueError:
            pass

    return 20.0