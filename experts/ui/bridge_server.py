"""Côté fenêtre : serveur WebSocket local (Qt) qui reçoit les évènements des serveurs LSP."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QHostAddress
from PySide6.QtWebSockets import QWebSocket, QWebSocketServer

from experts.core.bridge import DEFAULT_PORT, check_hello, parse_event


class BridgeServer(QObject):
    event_received = Signal(dict)
    clients_changed = Signal(int)

    def __init__(self, token: str, port: int = DEFAULT_PORT, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._token = token
        self._server = QWebSocketServer("experts", QWebSocketServer.SslMode.NonSecureMode, self)
        self._pending: set[QWebSocket] = set()
        self._clients: list[QWebSocket] = []
        self.active: QWebSocket | None = None  # dernier serveur LSP qui a signalé un fichier
        if not self._server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), port):
            raise OSError(f"Port {port} occupé : une autre fenêtre du compagnon tourne déjà ?")
        self._server.newConnection.connect(self._on_new_connection)

    @property
    def port(self) -> int:
        return int(self._server.serverPort())

    def _on_new_connection(self) -> None:
        sock = self._server.nextPendingConnection()
        if sock is None:
            return
        if sock.origin():  # navigateur : refus (protection contre une page web malveillante)
            sock.close()
            return
        self._pending.add(sock)
        sock.textMessageReceived.connect(lambda raw, s=sock: self._on_message(s, raw))
        sock.disconnected.connect(lambda s=sock: self._on_disconnected(s))

    def _on_message(self, sock: QWebSocket, raw: str) -> None:
        if sock in self._pending:
            self._pending.discard(sock)
            if not check_hello(raw, self._token):
                sock.close()
                return
            self._clients.append(sock)
            self.clients_changed.emit(len(self._clients))
            return
        if sock not in self._clients:
            return
        event = parse_event(raw)
        if event is None:
            return
        if event["type"] == "focus":
            self.active = sock
        self.event_received.emit(event)

    def _on_disconnected(self, sock: QWebSocket) -> None:
        self._pending.discard(sock)
        if sock in self._clients:
            self._clients.remove(sock)
            self.clients_changed.emit(len(self._clients))
        if self.active is sock:
            self.active = self._clients[-1] if self._clients else None
        sock.deleteLater()

    def send(self, message: dict[str, Any]) -> bool:
        """Envoie au serveur LSP actif (utilisé à l'étape 7 pour les questions)."""
        if self.active is None:
            return False
        self.active.sendTextMessage(json.dumps(message))
        return True

    def close(self) -> None:
        for sock in list(self._clients):
            sock.close()
        self._server.close()
