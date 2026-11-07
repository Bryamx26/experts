"""Liaison locale entre le serveur LSP et la fenêtre du compagnon (WebSocket).

Choix d'architecture : la fenêtre est le serveur WebSocket, chaque serveur LSP
est un client. Un éditeur lance souvent plusieurs serveurs LSP (un par projet),
alors qu'il n'y a qu'une fenêtre : elle les écoute tous et suit le dernier actif.

Sécurité : un WebSocket sur 127.0.0.1 est joignable par n'importe quel
processus local et même par une page web ouverte dans le navigateur. Donc :
- écoute sur 127.0.0.1 seulement ;
- jeton secret aléatoire dans un fichier lisible par toi seul (0600), exigé
  dans le premier message, sinon la connexion est fermée ;
- toute connexion qui annonce un en-tête Origin (navigateur) est refusée.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TOKEN_FILE = "ws-token"
PROTOCOL_VERSION = 1
log = logging.getLogger("experts.bridge")

# Évènements LSP → fenêtre : focus, state, diagnostics, suggestion (+ answer, confirm à l'étape 7).
EVENT_TYPES = frozenset({"focus", "state", "diagnostics", "suggestion", "answer", "confirm"})
STATES = frozenset({"idle", "think", "alert", "suggest"})

Event = dict[str, Any]


def token_path(data_dir: Path) -> Path:
    return data_dir / TOKEN_FILE


def create_token(data_dir: Path) -> str:
    """Nouveau jeton à chaque démarrage de la fenêtre, fichier en 0600."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = token_path(data_dir)
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    path.chmod(0o600)
    return token


def read_token(data_dir: Path) -> str | None:
    try:
        return token_path(data_dir).read_text().strip() or None
    except OSError:
        return None


def hello(token: str, role: str = "lsp") -> str:
    return json.dumps({"type": "hello", "token": token, "role": role, "version": PROTOCOL_VERSION})


def check_hello(raw: str, token: str) -> bool:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(msg, dict)
        and msg.get("type") == "hello"
        and secrets.compare_digest(str(msg.get("token", "")), token)
    )


def parse_event(raw: str) -> Event | None:
    """Message valide et connu, ou None (les autres sont ignorés, jamais exécutés)."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict) or msg.get("type") not in EVENT_TYPES:
        return None
    if msg["type"] == "state" and msg.get("state") not in STATES:
        return None
    return msg


class BridgeClient:
    """Côté LSP : envoie les évènements à la fenêtre, se reconnecte si elle redémarre.

    `emit()` ne bloque jamais l'éditeur : sans fenêtre, les évènements sont perdus.
    """

    def __init__(
        self,
        data_dir: Path,
        port: int = DEFAULT_PORT,
        on_message: Callable[[Event], Awaitable[None]] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._url = f"ws://{HOST}:{port}"
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self._on_message = on_message
        self._task: asyncio.Task[None] | None = None
        self.connected = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())

    def emit(self, event: Event) -> None:
        if not self.connected.is_set():
            return
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(json.dumps(event))

    async def _run(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import WebSocketException

        delay = 1.0
        while True:
            token = read_token(self._data_dir)
            if token:
                try:
                    # proxy=None : ne jamais faire passer ce trafic local par un proxy HTTP(S).
                    async with connect(self._url, proxy=None, open_timeout=3) as ws:
                        await ws.send(hello(token))
                        self.connected.set()
                        delay = 1.0
                        await self._pump(ws)
                except (OSError, WebSocketException, TimeoutError):
                    pass
                finally:
                    self.connected.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def _pump(self, ws: Any) -> None:
        async def sender() -> None:
            while True:
                await ws.send(await self._queue.get())

        async def receiver() -> None:
            async for raw in ws:
                if self._on_message is not None and isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict):
                        await self._on_message(msg)

        tasks = [asyncio.ensure_future(sender()), asyncio.ensure_future(receiver())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
