"""ULTRON launcher.

Run this from the project root with:  python run.py
It adds the 'src' folder to Python's search path and starts ULTRON.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ultron.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
