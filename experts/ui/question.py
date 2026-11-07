"""Étape 7 : panneau de question qui s'ouvre quand on clique sur l'expert.

D'après la maquette : en-tête « <Expert> · pose ta question », puce de contexte
(fichier, fonction, lignes), fil de discussion, sources (RAG, SearXNG),
boutons « Insérer (voir le diff) » et « Copier », champ pour enchaîner.
Le panneau s'ouvre à gauche de l'expert, au bord de l'écran, pour ne pas
masquer le code. « Insérer » n'écrit rien : il demande d'abord le diff.
"""

from __future__ import annotations

import re
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

PANEL_WIDTH = 420
_CODE_RE = re.compile(r"```[\w.+-]*\n(.*?)```", re.DOTALL)

STYLE = """
#panel { background: rgba(30, 32, 38, 245); border: 1px solid rgba(255,255,255,45); border-radius: 14px; }
QLabel { color: #e8e8e8; }
#title { font-weight: bold; color: #f5d04a; font-size: 14px; }
#subtitle { color: #a8a8a8; }
#chip { background: rgba(70,110,180,70); color: #b9d3ff; border-radius: 8px; padding: 3px 8px; }
#source { background: rgba(255,255,255,18); color: #cfcfcf; border-radius: 8px; padding: 2px 8px; }
QTextBrowser { background: transparent; border: none; color: #e8e8e8; }
QLineEdit { background: rgba(0,0,0,90); color: #eee; border: 1px solid #4a7fd1;
            border-radius: 8px; padding: 6px; }
QPushButton { color: #eee; background: rgba(255,255,255,20); border: 1px solid rgba(255,255,255,50);
              border-radius: 6px; padding: 5px 12px; }
QPushButton#primary { background: #1f6fd1; border: none; font-weight: bold; }
QPushButton#close { background: transparent; border: none; font-size: 14px; padding: 0 4px; }
QScrollBar:horizontal, QScrollBar:vertical { background: transparent; height: 8px; width: 8px; }
QScrollBar::handle { background: rgba(255,255,255,60); border-radius: 4px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
"""


def last_code_block(markdown: str) -> str | None:
    blocks = _CODE_RE.findall(markdown)
    return blocks[-1] if blocks else None


class QuestionPanel(QFrame):
    ask = Signal(str)
    insert_requested = Signal(str)
    confirm_answered = Signal(str, bool)
    closed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(STYLE)
        self.setFixedWidth(PANEL_WIDTH)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.title = QLabel("Python", self)
        self.title.setObjectName("title")
        subtitle = QLabel("· pose ta question", self)
        subtitle.setObjectName("subtitle")
        close = QPushButton("✕", self)
        close.setObjectName("close")
        close.clicked.connect(self._close)
        header.addWidget(self.title)
        header.addWidget(subtitle)
        header.addStretch(1)
        header.addWidget(close)
        root.addLayout(header)

        self.chip = QLabel("Contexte : fichier actif", self)
        self.chip.setObjectName("chip")
        root.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignLeft)

        self.transcript = QTextBrowser(self)
        self.transcript.setOpenLinks(False)  # aucun lien suivi depuis une réponse du modèle
        self.transcript.setMinimumHeight(220)
        root.addWidget(self.transcript, 1)

        self.sources_row = QHBoxLayout()
        root.addLayout(self.sources_row)

        self.confirm_box = QFrame(self)
        cb = QVBoxLayout(self.confirm_box)
        cb.setContentsMargins(0, 0, 0, 0)
        self.confirm_label = QLabel(self.confirm_box)
        self.confirm_label.setWordWrap(True)
        cbuttons = QHBoxLayout()
        self.allow = QPushButton("Autoriser", self.confirm_box)
        self.allow.setObjectName("primary")
        self.deny = QPushButton("Refuser", self.confirm_box)
        cbuttons.addWidget(self.allow)
        cbuttons.addWidget(self.deny)
        cbuttons.addStretch(1)
        cb.addWidget(self.confirm_label)
        cb.addLayout(cbuttons)
        self.confirm_box.hide()
        root.addWidget(self.confirm_box)
        self._confirm_id = ""
        self.allow.clicked.connect(lambda: self._answer_confirm(True))
        self.deny.clicked.connect(lambda: self._answer_confirm(False))

        actions = QHBoxLayout()
        self.insert_btn = QPushButton("Insérer (voir le diff)", self)
        self.insert_btn.setObjectName("primary")
        self.copy_btn = QPushButton("Copier", self)
        actions.addWidget(self.insert_btn)
        actions.addWidget(self.copy_btn)
        actions.addStretch(1)
        root.addLayout(actions)
        self.insert_btn.clicked.connect(self._insert)
        self.copy_btn.clicked.connect(self._copy)

        self.input = QLineEdit(self)
        self.input.setPlaceholderText("Pose ta question…   (Entrée)")
        self.input.returnPressed.connect(self._submit)
        root.addWidget(self.input)

        self._markdown = ""
        self._code: str | None = None
        self._busy = False
        self._refresh_actions()

    # --- état ---
    def set_companion(self, display_name: str) -> None:
        self.title.setText(display_name)

    def set_context(self, label: str) -> None:
        self.chip.setText(f"Contexte : {label}")

    def _refresh_actions(self) -> None:
        self.insert_btn.setEnabled(self._code is not None and not self._busy)
        self.copy_btn.setEnabled(self._code is not None)
        self.input.setEnabled(not self._busy)

    def _render(self) -> None:
        self.transcript.setMarkdown(self._markdown)
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    # --- interactions ---
    def _submit(self) -> None:
        question = self.input.text().strip()
        if not question or self._busy:
            return
        self.input.clear()
        self._markdown += f"\n\n**Toi :** {question}\n\n*…l'expert réfléchit…*"
        self._busy = True
        self._render()
        self._refresh_actions()
        self.ask.emit(question)

    def show_answer(self, text: str, sources: list[str], context: str | None) -> None:
        self._markdown = self._markdown.replace("\n\n*…l'expert réfléchit…*", "")
        self._markdown += f"\n\n{text}"
        self._code = last_code_block(text)
        self._busy = False
        if context:
            self.set_context(context)
        while (item := self.sources_row.takeAt(0)) is not None:
            if (w := item.widget()) is not None:
                w.deleteLater()
        if sources:
            label = QLabel("Sources :", self)
            label.setObjectName("subtitle")
            self.sources_row.addWidget(label)
            for s in sources:
                chip = QLabel(escape(s if len(s) <= 28 else s[:27] + "…"), self)
                chip.setToolTip(s)
                chip.setObjectName("source")
                self.sources_row.addWidget(chip)
            self.sources_row.addStretch(1)
        self._render()
        self._refresh_actions()
        self.input.setFocus()

    def ask_confirmation(self, cid: str, command: str) -> None:
        self._confirm_id = cid
        self.confirm_label.setText(f"L'expert veut exécuter :\n<code>{escape(command)}</code>")
        self.confirm_label.setTextFormat(Qt.TextFormat.RichText)
        self.confirm_box.show()

    def _answer_confirm(self, ok: bool) -> None:
        self.confirm_box.hide()
        self.confirm_answered.emit(self._confirm_id, ok)

    def _insert(self) -> None:
        if self._code is not None:
            self.insert_requested.emit(self._code)

    def _copy(self) -> None:
        if self._code is not None:
            QGuiApplication.clipboard().setText(self._code)

    def _close(self) -> None:
        if self.confirm_box.isVisible():
            self._answer_confirm(False)  # fermer = refuser
        self.hide()
        self.closed.emit()


class DiffDialog(QDialog):
    """Montre le diff avant toute écriture dans l'éditeur."""

    def __init__(self, filename: str, diff: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Insérer dans {filename} ?")
        self.resize(640, 420)
        layout = QVBoxLayout(self)
        view = QPlainTextEdit(self)
        view.setReadOnly(True)
        view.setStyleSheet("font-family: monospace;")
        view.appendHtml(
            "<pre>"
            + "".join(
                f'<span style="color:{color_of(line)}">{escape(line)}</span>'
                for line in diff.splitlines(keepends=True)
            )
            + "</pre>"
        )
        layout.addWidget(view)
        buttons = QDialogButtonBox(self)
        apply_btn = buttons.addButton("Appliquer", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        apply_btn.setDefault(False)  # Entrée ne doit pas appliquer par accident
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def color_of(line: str) -> str:
    if line.startswith("+") and not line.startswith("+++"):
        return "#6fcf6f"
    if line.startswith("-") and not line.startswith("---"):
        return "#ef6b6b"
    if line.startswith("@@"):
        return "#7fb2ff"
    return "#cccccc"
