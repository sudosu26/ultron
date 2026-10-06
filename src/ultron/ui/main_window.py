"""Main ULTRON window — chat, sidebar, multi-key, and voice."""

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QStatusBar,
    QTextEdit, QVBoxLayout, QWidget,
)

from ultron.core.chat_engine import ChatEngine
from ultron.credentials.vault import (
    get_api_keys, get_credential, set_api_keys, set_credential,
)
from ultron.database.queries import (
    add_message, count_messages, create_conversation,
    get_conversation, get_setting, list_conversations, list_messages,
    set_setting, update_conversation_title,
)
from ultron.utils.logging import setup_logging
from ultron.voice import stt, tts
from ultron.voice.recorder import Recorder

logger = setup_logging()

_PROVIDER = "gemini"
_GROQ_KEY = "groq_api_key"


_DARK_STYLE = """
QMainWindow, QWidget { background-color: #0d1117; color: #e6edf3; }
QTextEdit, QPlainTextEdit { background-color: #161b22; border: 1px solid #30363d;
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
QPushButton#newChatBtn { background-color: #1f6feb; }
QPushButton#newChatBtn:hover { background-color: #388bfd; }
QPushButton#exportBtn { background-color: #21262d; color: #e6edf3; }
QPushButton#exportBtn:hover { background-color: #30363d; }
QPushButton#micBtn { background-color: #21262d; color: #e6edf3;
                     padding: 10px 14px; font-size: 18px; }
QPushButton#micBtn:hover { background-color: #30363d; }
QPushButton#micBtn[recording="true"] { background-color: #da3633; }
QPushButton#speakBtn { background-color: #21262d; color: #e6edf3; }
QPushButton#speakBtn:hover { background-color: #30363d; }
QPushButton#speakBtn:checked { background-color: #1f6feb; color: #ffffff; }
QComboBox { background-color: #161b22; color: #e6edf3;
            border: 1px solid #30363d; border-radius: 6px; padding: 6px; }
QListWidget { background-color: #161b22; border: 1px solid #30363d;
              border-radius: 8px; padding: 4px; font-size: 13px; }
QListWidget::item { padding: 8px; border-radius: 4px; }
QListWidget::item:selected { background-color: #1f6feb; color: #ffffff; }
QListWidget::item:hover { background-color: #21262d; }
QStatusBar { background-color: #161b22; color: #8b949e; }
QLabel { font-size: 13px; }
"""


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class StreamWorker(QThread):
    chunk_received = Signal(str)
    finished_ok = Signal(str)
    error_occurred = Signal(str)
    cancelled = Signal()

    def __init__(self, engine: ChatEngine, api_keys: list[str]) -> None:
        super().__init__()
        self.engine = engine
        self.api_keys = api_keys
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[str] | None = None
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()
        loop = self._loop
        task = self._task
        if loop is not None and task is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._consume())
        self._task = task
        try:
            if self._cancel_requested.is_set():
                task.cancel()
            full = loop.run_until_complete(task)
            self.finished_ok.emit(full)
        except asyncio.CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            logger.error("Stream worker error: %s", type(exc).__name__)
            self.error_occurred.emit(str(exc))
        finally:
            self._task = None
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                self._loop = None

    async def _consume(self) -> str:
        text = ""
        async for chunk in self.engine.stream_response(self.api_keys):
            self.chunk_received.emit(chunk)
            text += chunk
        return text


class VoiceWorker(QThread):
    """Transcribes a WAV file in the background."""
    transcript_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, wav_path: str, api_key: str) -> None:
        super().__init__()
        self.wav_path = wav_path
        self.api_key = api_key

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            text = loop.run_until_complete(
                stt.transcribe(self.wav_path, self.api_key)
            )
            self.transcript_ready.emit(text)
        except Exception as exc:
            logger.error("Voice worker error: %s", type(exc).__name__)
            self.error_occurred.emit(str(exc))
        finally:
            loop.close()
        try:
            Path(self.wav_path).unlink(missing_ok=True)
        except Exception:
            pass


class TTSWorker(QThread):
    """Synthesizes and plays TTS in the background."""
    error_occurred = Signal(str)

    def __init__(self, text: str, voice: str) -> None:
        super().__init__()
        self.text = text
        self.voice = voice

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        path: str | None = None
        try:
            path = loop.run_until_complete(
                tts.synthesize_to_file(self.text, self.voice)
            )
        except Exception as exc:
            logger.error("TTS synth error: %s", type(exc).__name__)
            self.error_occurred.emit(f"TTS synthesis failed: {exc}")
            return
        finally:
            loop.close()

        try:
            tts.play_file_sync(path)
        except Exception as exc:
            logger.error("TTS playback error: %s", type(exc).__name__)
            self.error_occurred.emit(f"TTS playback failed: {exc}")
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ULTRON — Settings")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        # ---- Gemini section ----
        gem_label = QLabel("<b>Gemini API keys</b> — one per line")
        layout.addWidget(gem_label)

        gem_info = QLabel(
            "Get free keys at aistudio.google.com/app/apikey. "
            "ULTRON rotates to the next key on rate limit."
        )
        gem_info.setWordWrap(True)
        layout.addWidget(gem_info)

        self.gemini_input = QPlainTextEdit()
        existing_gem = get_api_keys(_PROVIDER)
        self.gemini_input.setPlainText("\n".join(existing_gem))
        self.gemini_input.setPlaceholderText("AIza...\nAIza...")
        self.gemini_input.setMinimumHeight(90)
        layout.addWidget(self.gemini_input)

        self.gem_count = QLabel(f"{len(existing_gem)} Gemini key(s) saved.")
        layout.addWidget(self.gem_count)

        # ---- Groq section ----
        groq_label = QLabel("<b>Groq API key</b> — for speech-to-text")
        layout.addWidget(groq_label)

        groq_info = QLabel(
            "Get a free key at console.groq.com/keys. Required for the "
            "microphone button."
        )
        groq_info.setWordWrap(True)
        layout.addWidget(groq_info)

        self.groq_input = QLineEdit()
        self.groq_input.setEchoMode(QLineEdit.EchoMode.Password)
        existing_groq = get_credential(_GROQ_KEY)
        if existing_groq:
            self.groq_input.setPlaceholderText(
                "(saved — type to replace, clear to remove)"
            )
        else:
            self.groq_input.setPlaceholderText("gsk_...")
        layout.addWidget(self.groq_input)

        # ---- TTS voice ----
        voice_label = QLabel("<b>Text-to-speech voice</b>")
        layout.addWidget(voice_label)

        voice_row = QHBoxLayout()
        self.voice_combo = QComboBox()
        for v in tts.AVAILABLE_VOICES:
            self.voice_combo.addItem(v)
        current = get_setting("tts_voice", tts.DEFAULT_VOICE) or tts.DEFAULT_VOICE
        if current in tts.AVAILABLE_VOICES:
            self.voice_combo.setCurrentText(current)
        voice_row.addWidget(QLabel("Voice:"))
        voice_row.addWidget(self.voice_combo, stretch=1)
        layout.addLayout(voice_row)

        # ---- Buttons ----
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self) -> None:
        # Gemini keys
        gem_raw = self.gemini_input.toPlainText()
        gem_keys = [ln.strip() for ln in gem_raw.splitlines() if ln.strip()]
        if gem_keys:
            set_api_keys(_PROVIDER, gem_keys)

        # Groq key (single)
        groq = self.groq_input.text().strip()
        if groq:
            set_credential(_GROQ_KEY, groq)

        # TTS voice
        set_setting("tts_voice", self.voice_combo.currentText())

        QMessageBox.information(self, "Saved", "Settings saved.")
        self.accept()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ULTRON — Personal AI Assistant")
        self.resize(1150, 740)
        self.setStyleSheet(_DARK_STYLE)

        self.engine = ChatEngine()
        self.worker: StreamWorker | None = None
        self.voice_worker: VoiceWorker | None = None
        self.tts_worker: TTSWorker | None = None
        self._workers: set[QThread] = set()
        self.recorder = Recorder()
        self.current_conv_id: str | None = None
        self.speak_responses: bool = bool(get_setting("speak", "0") == "1")

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # --- Sidebar ---
        sidebar = QVBoxLayout()
        sidebar.setSpacing(8)

        self.new_chat_btn = QPushButton("+ New Chat")
        self.new_chat_btn.setObjectName("newChatBtn")
        self.new_chat_btn.clicked.connect(self._new_conversation)
        sidebar.addWidget(self.new_chat_btn)

        self.conv_list = QListWidget()
        self.conv_list.itemSelectionChanged.connect(self._on_sidebar_selection)
        sidebar.addWidget(self.conv_list, stretch=1)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(240)
        outer.addWidget(sidebar_widget)

        # --- Main column ---
        main_col = QVBoxLayout()
        main_col.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("ULTRON")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()

        self.status_label = QLabel("● Not configured")
        header.addWidget(self.status_label)

        self.speak_btn = QPushButton("🔊 Speak")
        self.speak_btn.setObjectName("speakBtn")
        self.speak_btn.setCheckable(True)
        self.speak_btn.setChecked(self.speak_responses)
        self.speak_btn.clicked.connect(self._on_speak_toggled)
        header.addWidget(self.speak_btn)

        self.export_btn = QPushButton("⬇ Export")
        self.export_btn.setObjectName("exportBtn")
        self.export_btn.clicked.connect(self._export_conversation)
        header.addWidget(self.export_btn)

        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.clicked.connect(self._open_settings)
        header.addWidget(self.settings_btn)

        main_col.addLayout(header)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 12))
        main_col.addWidget(self.chat_display, stretch=1)

        input_row = QHBoxLayout()

        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setObjectName("micBtn")
        self.mic_btn.setToolTip(
            "Push and hold to talk. Release to send."
        )
        self.mic_btn.pressed.connect(self._on_mic_pressed)
        self.mic_btn.released.connect(self._on_mic_released)
        input_row.addWidget(self.mic_btn)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(
            "Type your message and press Enter…"
        )
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

        main_col.addLayout(input_row)

        main_widget = QWidget()
        main_widget.setLayout(main_col)
        outer.addWidget(main_widget, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        QShortcut(QKeySequence("Ctrl+,"), self, self._open_settings)
        QShortcut(QKeySequence("Escape"), self, self._stop_stream)
        QShortcut(QKeySequence("Ctrl+N"), self, self._new_conversation)

        self._refresh_status()
        self._load_or_create_conversation()

    # ------------------------------------------------------------------
    # Status / settings
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        n = len(get_api_keys(_PROVIDER))
        if n == 0:
            self.status_label.setText("● Not configured")
            self.status_label.setStyleSheet("color: #f85149;")
        elif n == 1:
            self.status_label.setText("● 1 key configured")
            self.status_label.setStyleSheet("color: #3fb950;")
        else:
            self.status_label.setText(f"● {n} keys configured")
            self.status_label.setStyleSheet("color: #3fb950;")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh_status()
            n = len(get_api_keys(_PROVIDER))
            has_groq = bool(get_credential(_GROQ_KEY))
            extra = " Voice ready." if has_groq else " Add a Groq key for voice."
            self._append_system(f"Saved. {n} Gemini key(s).{extra}")

    # ------------------------------------------------------------------
    # Speak toggle
    # ------------------------------------------------------------------

    def _on_speak_toggled(self) -> None:
        self.speak_responses = self.speak_btn.isChecked()
        set_setting("speak", "1" if self.speak_responses else "0")
        if not self.speak_responses:
            tts.stop_playback()

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _load_or_create_conversation(self) -> None:
        convs = list_conversations()
        if convs:
            self._open_conversation(convs[0]["id"])
        else:
            self._new_conversation()

    def _new_conversation(self) -> None:
        self.current_conv_id = create_conversation()
        self.engine.reset()
        self.chat_display.clear()
        self._refresh_sidebar()

    def _open_conversation(self, conv_id: str) -> None:
        self.current_conv_id = conv_id
        stored = list_messages(conv_id)
        self.engine.load_history(stored)
        self.chat_display.clear()
        for m in stored:
            if m["role"] == "user":
                self._append_user(m["content"])
            elif m["role"] == "assistant":
                self._append_assistant_complete(m["content"])
        self._refresh_sidebar()

    def _refresh_sidebar(self) -> None:
        self.conv_list.blockSignals(True)
        self.conv_list.clear()
        for c in list_conversations():
            item = QListWidgetItem(c["title"])
            item.setData(Qt.ItemDataRole.UserRole, c["id"])
            self.conv_list.addItem(item)
        for i in range(self.conv_list.count()):
            item = self.conv_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self.current_conv_id:
                self.conv_list.setCurrentItem(item)
                break
        self.conv_list.blockSignals(False)

    def _on_sidebar_selection(self) -> None:
        items = self.conv_list.selectedItems()
        if not items:
            return
        conv_id = items[0].data(Qt.ItemDataRole.UserRole)
        if conv_id != self.current_conv_id:
            self._open_conversation(conv_id)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

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
            '<p style="color:#3fb950;margin-top:12px;"><b>ULTRON:</b> </p>'
        )

    def _append_assistant_chunk(self, text: str) -> None:
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(text)
        self.chat_display.ensureCursorVisible()

    def _append_assistant_complete(self, text: str) -> None:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.chat_display.append(
            f'<p style="color:#3fb950;margin-top:12px;">'
            f'<b>ULTRON:</b> {safe}</p>'
        )

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _send_message(self) -> None:
        text = self.input_field.text().strip()
        if not text:
            return
        if self.worker and self.worker.isRunning():
            self._append_system(
                "A reply is already in progress. Wait for it to finish "
                "before sending another message."
            )
            return

        api_keys = get_api_keys(_PROVIDER)
        if not api_keys:
            QMessageBox.warning(
                self, "API Keys Required",
                "Please open Settings (⚙) and add at least one Gemini API key.",
            )
            return

        if not self.current_conv_id:
            self._new_conversation()

        if count_messages(self.current_conv_id) == 0:
            title = text.strip()
            if len(title) > 40:
                title = title[:40].rstrip() + "…"
            update_conversation_title(self.current_conv_id, title)
            self._refresh_sidebar()

        add_message(self.current_conv_id, "user", text)

        self.input_field.clear()
        self._append_user(text)
        self.engine.add_user_message(text)

        self._append_assistant_start()
        self._set_busy(True)

        self.worker = StreamWorker(self.engine, api_keys)
        self.worker.chunk_received.connect(self._append_assistant_chunk)
        self.worker.finished_ok.connect(self._on_stream_finished)
        self.worker.error_occurred.connect(self._on_stream_error)
        self.worker.cancelled.connect(self._on_stream_cancelled)
        self._track_worker(self.worker)
        self.worker.start()

    def _on_stream_finished(self, full_response: str) -> None:
        if self.current_conv_id and full_response:
            add_message(self.current_conv_id, "assistant", full_response)
        self._set_busy(False)
        self._refresh_sidebar()
        self.statusBar().showMessage("Ready")

        if self.speak_responses and full_response:
            self._speak(full_response)

    def _on_stream_error(self, message: str) -> None:
        self._set_busy(False)
        self._append_system(f"Error: {message}")
        self.statusBar().showMessage("Error — see message above")

    def _on_stream_cancelled(self) -> None:
        self._set_busy(False)
        self._append_system("(stopped by user)")
        self.statusBar().showMessage("Stopped")

    def _stop_stream(self) -> None:
        tts.stop_playback()
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("Stopping response…")
            self.worker.request_cancel()

    def _set_busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.input_field.setEnabled(not busy)
        if busy:
            self.statusBar().showMessage("ULTRON is thinking…")

    # ------------------------------------------------------------------
    # Voice
    # ------------------------------------------------------------------

    def _speak(self, text: str) -> None:
        voice = get_setting("tts_voice", tts.DEFAULT_VOICE) or tts.DEFAULT_VOICE
        # Stop any prior TTS before starting a new one.
        tts.stop_playback()
        self.tts_worker = TTSWorker(text, voice)
        self.tts_worker.error_occurred.connect(
            lambda msg: self._append_system(f"TTS: {msg}")
        )
        self._track_worker(self.tts_worker)
        self.tts_worker.start()

    def _on_mic_pressed(self) -> None:
        # Interrupt any TTS currently playing.
        tts.stop_playback()

        groq_key = get_credential(_GROQ_KEY)
        if not groq_key:
            self._append_system(
                "No Groq API key. Add one in Settings (⚙) to use the mic."
            )
            return

        try:
            self.recorder.start()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Microphone Error", str(exc))
            return

        self.mic_btn.setProperty("recording", "true")
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)
        self.statusBar().showMessage("🔴 Recording… release to send")

    def _on_mic_released(self) -> None:
        wav_path = self.recorder.stop()
        self.mic_btn.setProperty("recording", "false")
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)

        if not wav_path:
            self.statusBar().showMessage("Ready")
            return

        groq_key = get_credential(_GROQ_KEY)
        if not groq_key:
            self.statusBar().showMessage("Ready")
            return

        self.statusBar().showMessage("Transcribing…")
        self.voice_worker = VoiceWorker(wav_path, groq_key)
        self.voice_worker.transcript_ready.connect(self._on_transcript_ready)
        self.voice_worker.error_occurred.connect(self._on_transcript_error)
        self._track_worker(self.voice_worker)
        self.voice_worker.start()

    def _on_transcript_ready(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.statusBar().showMessage("No speech detected.")
            return
        self.input_field.setText(text)
        self._send_message()

    def _on_transcript_error(self, message: str) -> None:
        self._append_system(f"Voice error: {message}")
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_conversation(self) -> None:
        if not self.current_conv_id:
            QMessageBox.information(self, "Export", "No conversation selected.")
            return
        conv = get_conversation(self.current_conv_id)
        messages = list_messages(self.current_conv_id)
        if not conv or not messages:
            QMessageBox.information(self, "Export", "This conversation is empty.")
            return
        suggested = "".join(
            c for c in conv["title"] if c.isalnum() or c in " -_"
        ).strip() or "conversation"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation",
            f"{suggested}.md",
            "Markdown (*.md);;Plain text (*.txt)",
        )
        if not file_path:
            return
        lines = [
            f"# {conv['title']}", "",
            f"*Exported from ULTRON on "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}*",
            "", "---", "",
        ]
        for m in messages:
            speaker = "You" if m["role"] == "user" else "ULTRON"
            lines.append(f"### {speaker}")
            lines.append("")
            lines.append(m["content"])
            lines.append("")
        try:
            Path(file_path).write_text("\n".join(lines), encoding="utf-8")
            QMessageBox.information(self, "Export", f"Saved to:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Failed", f"Could not write file:\n{exc}"
            )

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------

    def _track_worker(self, worker: QThread) -> None:
        self._workers.add(worker)
        worker.finished.connect(self._on_worker_finished)

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if isinstance(worker, QThread):
            self._workers.discard(worker)

    def closeEvent(self, event) -> None:
        if any(worker.isRunning() for worker in self._workers):
            self.statusBar().showMessage(
                "A background task is still running. Stop the response "
                "or wait for it to finish before closing."
            )
            event.ignore()
            return
        try:
            tts.stop_playback()
        except Exception:
            pass
        try:
            self.recorder.cancel()
        except Exception:
            pass
        super().closeEvent(event)