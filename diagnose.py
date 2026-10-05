"""ULTRON diagnostic: test Gemini API key and model availability.

Run with:  python diagnose.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import httpx

from ultron.credentials.vault import get_credential

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

CANDIDATE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash",
    "gemini-3.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


def mask(key: str) -> str:
    """Return a safe, partial view of the key (never the full key)."""
    if not key:
        return "(empty)"
    if len(key) < 12:
        return "(too short)"
    return key[:6] + "..." + key[-4:] + f" (length {len(key)})"


async def main() -> None:
    api_key = get_credential("gemini_api_key")

    print("=" * 60)
    print("ULTRON GEMINI DIAGNOSTIC")
    print("=" * 60)
    print(f"API key in vault: {mask(api_key) if api_key else 'NONE — not saved'}")
    print()

    if not api_key:
        print("No API key saved. Open ULTRON Settings (⚙) and save a key first.")
        return

    # --- Test 1: list models ---
    print("-" * 60)
    print("TEST 1: List models (proves the API key is valid)")
    print("-" * 60)
    url = f"{GEMINI_BASE}/models?key={api_key}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(url)
            print(f"HTTP status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                models = data.get("models", [])
                print(f"Models returned: {len(models)}")
                print()
                print("Available model names (first 40):")
                for entry in models[:40]:
                    name = entry.get("name", "").replace("models/", "")
                    methods = entry.get("supportedGenerationMethods", [])
                    chat_ok = "generateContent" in methods
                    marker = "  ✅" if chat_ok else "  ❌"
                    print(f"{marker} {name}")
                print()
                usable = [
                    e.get("name", "").replace("models/", "")
                    for e in models
                    if "generateContent" in e.get("supportedGenerationMethods", [])
                ]
                print(f"Usable for chat: {len(usable)}")
                if usable:
                    print(f"First usable model: {usable[0]}")
            else:
                print("Response body (truncated):")
                print(r.text[:500])
        except Exception as exc:
            print(f"EXCEPTION: {type(exc).__name__}: {exc}")

    print()

    # --- Test 2: try each candidate model with a tiny message ---
    print("-" * 60)
    print("TEST 2: Try each candidate model with a 1-word prompt")
    print("-" * 60)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in CANDIDATE_MODELS:
            url = f"{GEMINI_BASE}/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hi."}]}
                ]
            }
            try:
                r = await client.post(url, json=payload)
                status = r.status_code
                note = ""
                if status == 200:
                    note = "✅ WORKS"
                elif status == 404:
                    note = "❌ 404 — model does not exist"
                elif status == 400:
                    body = r.text.lower()
                    if "api key" in body:
                        note = "❌ 400 — API key problem"
                    else:
                        note = "❌ 400 — bad request"
                elif status == 403:
                    note = "❌ 403 — permission denied (key issue?)"
                elif status == 429:
                    note = "⚠️ 429 — rate limited (but key works)"
                else:
                    note = f"⚠️ {status}"

                print(f"  {model:30s}  HTTP {status:3d}  {note}")

                if status not in (200, 404, 429):
                    # Show the error body for unexpected codes.
                    try:
                        err = r.json().get("error", {})
                        msg = err.get("message", r.text[:200])
                        print(f"      → {msg[:200]}")
                    except Exception:
                        print(f"      → {r.text[:200]}")
            except Exception as exc:
                print(f"  {model:30s}  EXCEPTION {type(exc).__name__}: {exc}")

    print()
    print("=" * 60)
    print("Diagnostic complete.")
    print("Please copy the ENTIRE output above and send it back.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
