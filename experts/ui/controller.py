"""Relie la fenêtre, le panneau de question (étape 7) et la liaison WebSocket."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtWidgets import QDialog

from experts.ui.question import DiffDialog, QuestionPanel
from experts.ui.window import CompanionWindow


class Sender(Protocol):
    def send(self, message: dict[str, Any]) -> bool: ...


class Controller(QObject):
    def __init__(self, window: CompanionWindow, bridge: Sender) -> None:
        super().__init__(window)
        self.window = window
        self.bridge = bridge
        self.panel = QuestionPanel()
        self.panel.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.panel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.panel.set_companion(window.current.display_name)
        window.sprite_clicked.connect(self.toggle_panel)
        self.panel.ask.connect(self.on_ask)
        self.panel.confirm_answered.connect(
            lambda cid, ok: self.bridge.send({"type": "confirm_reply", "id": cid, "ok": ok})
        )
        self.panel.insert_requested.connect(self.on_insert)
        self.panel.closed.connect(lambda: window.sprite.set_selected(False))
        # Remplaçable dans les tests ; en vrai, boîte de dialogue modale.
        self.confirm_diff: Callable[[DiffDialog], bool] = lambda d: d.exec() == QDialog.DialogCode.Accepted

    # --- panneau ---
    def toggle_panel(self) -> None:
        if self.panel.isVisible():
            self.panel.hide()
            self.window.sprite.set_selected(False)
            return
        self.window.sprite.set_selected(True)  # halo « je t'écoute », comme sur la maquette
        self.panel.adjustSize()
        anchor = self.window.sprite.mapToGlobal(QPoint(0, self.window.sprite.height()))
        self.panel.move(anchor.x() - self.panel.width() - 8, anchor.y() - self.panel.height())
        self.panel.show()
        self.panel.raise_()
        self.panel.input.setFocus()

    def on_ask(self, question: str) -> None:
        if not self.bridge.send({"type": "ask", "id": uuid.uuid4().hex, "question": question}):
            self.panel.show_answer(
                "Aucun éditeur connecté. Ouvre un fichier dans un éditeur qui utilise `experts-lsp`.",
                [],
                None,
            )

    def on_insert(self, code: str) -> None:
        self.bridge.send({"type": "insert_preview", "id": uuid.uuid4().hex, "code": code})

    # --- évènements venant du LSP ---
    def handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "focus":
            self.window.handle_event(event)
            self.panel.set_companion(self.window.current.display_name)
            self.panel.set_context(Path(str(event.get("file", ""))).name)
        elif kind == "answer":
            spec = self.window.companions.get(str(event.get("companion")))
            if spec is not None:
                self.panel.set_companion(spec.display_name)
            sources = [str(s) for s in event.get("sources") or []]
            self.panel.show_answer(str(event.get("text", "")), sources, event.get("context"))
        elif kind == "confirm":
            if not self.panel.isVisible():
                self.toggle_panel()
            self.panel.ask_confirmation(str(event.get("id")), str(event.get("command", "")))
        elif kind == "diff":
            if event.get("error") or not event.get("diff"):
                self.window.say(str(event.get("error") or "Rien à insérer : le code est identique."), 5000)
                return
            dialog = DiffDialog(str(event.get("file", "")), str(event["diff"]), self.panel)
            if self.confirm_diff(dialog):
                self.bridge.send({"type": "insert_apply", "id": event.get("id")})
        else:
            self.window.handle_event(event)
