"""Encrypted credential vault.

API keys are encrypted with a key stored in Windows Credential Manager.
The encrypted blob is written to a file on disk.

Supports multiple API keys per provider (stored as a JSON list under
'<provider>__keys'). Automatically migrates from the older single-key
format ('<provider>_api_key') on first read.
"""

import json
from pathlib import Path
from cryptography.fernet import Fernet
import keyring

from ultron.utils.paths import CONFIG_DIR
from ultron.utils.logging import setup_logging

logger = setup_logging()

_SERVICE_NAME = "ULTRON"
_KEY_NAME = "vault_master_key"
_VAULT_FILE = CONFIG_DIR / "vault.enc"


def _get_or_create_master_key() -> bytes:
    existing = keyring.get_password(_SERVICE_NAME, _KEY_NAME)
    if existing:
        return existing.encode("utf-8")
    new_key = Fernet.generate_key()
    keyring.set_password(_SERVICE_NAME, _KEY_NAME, new_key.decode("utf-8"))
    logger.info("Created new vault master key in Windows Credential Manager.")
    return new_key


def _load_vault() -> dict:
    if not _VAULT_FILE.exists():
        return {}
    try:
        master_key = _get_or_create_master_key()
        fernet = Fernet(master_key)
        encrypted = _VAULT_FILE.read_bytes()
        decrypted = fernet.decrypt(encrypted)
        return json.loads(decrypted.decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to decrypt vault: %s", type(exc).__name__)
        return {}


def _save_vault(data: dict) -> None:
    master_key = _get_or_create_master_key()
    fernet = Fernet(master_key)
    plaintext = json.dumps(data).encode("utf-8")
    encrypted = fernet.encrypt(plaintext)
    _VAULT_FILE.write_bytes(encrypted)
    logger.info("Vault saved (%d entries).", len(data))


# ---------------------------------------------------------------------------
# Single-value (kept for backward compatibility with earlier code)
# ---------------------------------------------------------------------------

def set_credential(name: str, value: str) -> None:
    data = _load_vault()
    data[name] = value
    _save_vault(data)
    logger.info("Credential '%s' stored.", name)


def get_credential(name: str) -> str | None:
    data = _load_vault()
    return data.get(name)


def delete_credential(name: str) -> None:
    data = _load_vault()
    if name in data:
        del data[name]
        _save_vault(data)
        logger.info("Credential '%s' deleted.", name)


def list_credentials() -> list[str]:
    return list(_load_vault().keys())


# ---------------------------------------------------------------------------
# Multi-key API (preferred for API providers)
# ---------------------------------------------------------------------------

def get_api_keys(provider: str) -> list[str]:
    """Return all API keys for a provider.

    Migrates from the single-key format ('<provider>_api_key') if the
    list is missing or empty. Never logs the keys themselves.
    """
    data = _load_vault()
    list_key = f"{provider}__keys"
    raw = data.get(list_key)
    if isinstance(raw, list) and raw:
        return [k for k in raw if isinstance(k, str) and k]
    # Migration: single key.
    single = data.get(f"{provider}_api_key")
    if isinstance(single, str) and single:
        return [single]
    return []


def set_api_keys(provider: str, keys: list[str]) -> None:
    """Replace the list of keys for a provider. Empty strings are dropped."""
    cleaned = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
    data = _load_vault()
    data[f"{provider}__keys"] = cleaned
    _save_vault(data)
    logger.info("Stored %d key(s) for provider '%s'.", len(cleaned), provider)


def add_api_key(provider: str, key: str) -> None:
    key = key.strip()
    if not key:
        return
    keys = get_api_keys(provider)
    if key not in keys:
        keys.append(key)
    set_api_keys(provider, keys)


def remove_api_key(provider: str, key: str) -> None:
    keys = [k for k in get_api_keys(provider) if k != key]
    set_api_keys(provider, keys)


def count_api_keys(provider: str) -> int:
    return len(get_api_keys(provider))