"""Main ULTRON window — chat interface with streaming responses."""

import asyncio

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel,
    QMessageBox, QDialog, QFormLayout, QDialogButtonBox,
    QStatusBar,
)

from ultron.core.chat_engine import ChatEngine
from ultron.credentials.vault import get_credential, set_credential, list_credentials
from ultron.utils.logging import setup_logging

logger = setup_logging()

_DARK_STYLE = """
QMainWindow { background-color: #0d1117; }
QWidget { background-color: #0d1117; color: #e6edf3; }
QTextEdit { background-color: #161b22; border: 1px solid #30363d;
            border-radius: 8px; padding: 10px; font-size: 14px; }
QLineEdit { background-color: #161b22; border: 1px solid #30363d;
            border-radius: 8px; padding: 10px; font-size: 14px; }
QPushButton { background-color: #238636; color: #ffffff; border: none;
              border-radius: 8px; padding: 10px 20px; font-size: 14px;
              font-weight: bold; }
QPushButton:hover { background-color: #2ea043; }
QPushButton:disabled { background-color: #21262d; color: #484f58; }
QPushButton#stopBtn { background-color: #da3633; }
QPushButton#stopBtn:hover { background-color: #f85149; }
QPushButton#settingsBtn { background-color: #21262d; color: #e6edf3; }
QPushButton#settingsBtn:hover { background-color: #30363d; }
QStatusBar { background-color: #161b22; color: #8b949e; }
QLabel { font-size: 13px; }
"""


class StreamWorker(QThread):
    """Runs the async streaming call in a background thread."""

    chunk_received = Signal(str)
    finished_ok = Signal()
    error_occurred = Signal(str)

    def __init__(self, engine: ChatEngine, api_key: str) -> None:
        super().__init__()
        self.engine = engine
        self.api_key = api_key

    def run(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._stream())
        except Exception as exc:
            logger.error("Stream worker error: %s", type(exc).__name__)
            self.error_occurred.emit(str(exc))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _stream(self) -> None:
        async for chunk in self.engine.stream_response(self.api_key):
            self.chunk_received.emit(chunk)
        self.finished_ok.emit()


class SettingsDialog(QDialog):
    """Dialog for entering and saving the AI provider API key."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ULTRON — Settings")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("Paste your Gemini API key here")

        existing = get_credential("gemini_api_key")
        if existing:
            self.key_input.setPlaceholderText("(saved — type to replace)")

        form.addRow("Gemini API Key:", self.key_input)
        layout.addLayout(form)

        note = QLabel(
            "Your key is encrypted and stored on this computer.\n"
            "It is never written to plain files or sent anywhere except Google."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self) -> None:
        key = self.key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Missing Key", "Please paste your API key.")
            return
        set_credential("gemini_api_key", key)
        QMessageBox.information(self, "Saved", "API key saved securely.")
        self.accept()


class MainWindow(QMainWindow):
    """ULTRON main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ULTRON — Personal AI Assistant")
        self.resize(900, 700)
        self.setStyleSheet(_DARK_STYLE)

        self.engine = ChatEngine()
        self.worker: StreamWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header row.
        header = QHBoxLayout()
        title = QLabel("ULTRON")
        title_font = QFont("Segoe UI", 20, QFont.Weight.Bold)
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch()

        self.status_label = QLabel("● Not configured")
        header.addWidget(self.status_label)

        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.clicked.connect(self._open_settings)
        header.addWidget(self.settings_btn)

        layout.addLayout(header)

        # Chat display.
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 12))
        layout.addWidget(self.chat_display, stretch=1)

        # Input row.
        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message and press Enter…")
        self.input_field.returnPressed.connect(self._send_message)
        input_row.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_stream)
        input_row.addWidget(self.stop_btn)

        layout.addLayout(input_row)

        # Status bar.
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        # Keyboard shortcuts.
        QShortcut(QKeySequence("Ctrl+,"), self, self._open_settings)
        QShortcut(QKeySequence("Escape"), self, self._stop_stream)

        self._refresh_status()

    def _refresh_status(self) -> None:
        if get_credential("gemini_api_key"):
            self.status_label.setText("● Gemini configured")
            self.status_label.setStyleSheet("color: #3fb950;")
        else:
            self.status_label.setText("● Not configured")
            self.status_label.setStyleSheet("color: #f85149;")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_status()
            self._append_system("API key saved. You can now send a message.")

    def _append_system(self, text: str) -> None:
        self.chat_display.append(
            f'<p style="color:#8b949e;font-style:italic;">{text}</p>'
        )

    def _append_user(self, text: str) -> None:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.chat_display.append(
            f'<p style="color:#58a6ff;margin-top:12px;"><b>You:</b> {safe}</p>'
        )

    def _append_assistant_start(self) -> None:
        self.chat_display.append(
            '<p style="color:#3fb950;margin-top:12px;"><b>ULTRON:</b> '
            '<span id="streaming"></span></p>'
        )

    def _append_assistant_chunk(self, text: str) -> None:
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(text)
        self.chat_display.ensureCursorVisible()

    def _send_message(self) -> None:
        text = self.input_field.text().strip()
        if not text:
            return

        api_key = get_credential("gemini_api_key")
        if not api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                "Please open Settings (⚙) and enter your Gemini API key first.",
            )
            return

        self.input_field.clear()
        self._append_user(text)
        self.engine.add_user_message(text)

        self._append_assistant_start()
        self._set_busy(True)

        self.worker = StreamWorker(self.engine, api_key)
        self.worker.chunk_received.connect(self._append_assistant_chunk)
        self.worker.finished_ok.connect(self._on_stream_finished)
        self.worker.error_occurred.connect(self._on_stream_error)
        self.worker.start()

    def _on_stream_finished(self) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Ready")

    def _on_stream_error(self, message: str) -> None:
        self._set_busy(False)
        self._append_system(f"Error: {message}")
        self.statusBar().showMessage("Error — see message above")

    def _stop_stream(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1000)
            self._append_system("(stopped by user)")
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.input_field.setEnabled(not busy)
        if busy:
            self.statusBar().showMessage("ULTRON is thinking…")