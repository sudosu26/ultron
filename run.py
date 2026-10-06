"""ULTRON launcher.

Run this from the project root with:  python run.py
It adds the 'src' folder to Python's search path and starts ULTRON.
"""

import subprocess
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent
_REQ_FILE = _SOURCE_ROOT / "requirements.txt"
if _REQ_FILE.exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(_REQ_FILE)])

_SRC = _SOURCE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ultron.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
