"""Encrypted credential vault.

API keys are encrypted with a key stored in Windows Credential Manager.
The encrypted blob is written to a file on disk.
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
    """Retrieve the master encryption key from Windows Credential Manager.

    If it does not exist, generate a new one and store it.
    """
    existing = keyring.get_password(_SERVICE_NAME, _KEY_NAME)
    if existing:
        return existing.encode("utf-8")
    new_key = Fernet.generate_key()
    keyring.set_password(_SERVICE_NAME, _KEY_NAME, new_key.decode("utf-8"))
    logger.info("Created new vault master key in Windows Credential Manager.")
    return new_key


def _load_vault() -> dict:
    """Load and decrypt the vault file. Returns an empty dict if no vault exists."""
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
    """Encrypt and write the vault file."""
    master_key = _get_or_create_master_key()
    fernet = Fernet(master_key)
    plaintext = json.dumps(data).encode("utf-8")
    encrypted = fernet.encrypt(plaintext)
    _VAULT_FILE.write_bytes(encrypted)
    logger.info("Vault saved (%d entries).", len(data))


def set_credential(name: str, value: str) -> None:
    """Store a credential (encrypted) in the vault."""
    data = _load_vault()
    data[name] = value
    _save_vault(data)
    logger.info("Credential '%s' stored.", name)


def get_credential(name: str) -> str | None:
    """Retrieve a credential from the vault. Returns None if not found."""
    data = _load_vault()
    return data.get(name)


def delete_credential(name: str) -> None:
    """Remove a credential from the vault."""
    data = _load_vault()
    if name in data:
        del data[name]
        _save_vault(data)
        logger.info("Credential '%s' deleted.", name)


def list_credentials() -> list[str]:
    """Return the names of stored credentials (not their values)."""
    return list(_load_vault().keys())