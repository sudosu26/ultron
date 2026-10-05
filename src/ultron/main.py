"""ULTRON entry point.

Run this file to launch the application.
"""

import sys

from PySide6.QtWidgets import QApplication

from ultron.ui.main_window import MainWindow
from ultron.utils.logging import setup_logging

logger = setup_logging()


def main() -> int:
    logger.info("ULTRON starting…")
    app = QApplication(sys.argv)
    app.setApplicationName("ULTRON")
    app.setOrganizationName("ULTRON")

    window = MainWindow()
    window.show()

    logger.info("ULTRON window shown.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())