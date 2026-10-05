"""Portable path resolution for ULTRON.

All paths are relative to the application root.
The application root is determined from the executable location,
never from a hardcoded drive letter.
"""

import sys
from pathlib import Path


def get_app_root() -> Path:
    """Return the ULTRON application root directory.

    When running from source (development): the folder containing 'src'.
    When running as a frozen executable (PyInstaller): the folder containing the .exe.
    """
    if getattr(sys, "frozen", False):
        # Running as a PyInstaller bundle.
        return Path(sys.executable).parent
    # Running from source.
    return Path(__file__).resolve().parents[3]


APP_ROOT: Path = get_app_root()
DATA_DIR: Path = APP_ROOT / "data"
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
ATTACHMENTS_DIR: Path = DATA_DIR / "attachments"
LOGS_DIR: Path = DATA_DIR / "logs"
CONFIG_DIR: Path = APP_ROOT / "config"

# Ensure directories exist at import time.
for _dir in (DATA_DIR, DOCUMENTS_DIR, ATTACHMENTS_DIR, LOGS_DIR, CONFIG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)