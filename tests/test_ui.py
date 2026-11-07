"""Étape 5 : fenêtre et liaison WebSocket, en mode Qt « offscreen » (sans écran)."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QCoreApplication, QElapsedTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from experts.core.bridge import (  # noqa: E402
    BridgeClient,
    check_hello,
    create_token,
    hello,
    parse_event,
    read_token,
)
from experts.core.companions import load_companions  # noqa: E402
from experts.ui.bridge_server import BridgeServer  # noqa: E402
from experts.ui.window import CompanionWindow  # noqa: E402


@pytest.fixture(scope="module")
def app() -> Iterator[QApplication]:
    instance = QApplication.instance() or QApplication([])
    assert isinstance(instance, QApplication)
    yield instance


def pump(until: object, timeout_ms: int = 5000) -> None:
    timer = QElapsedTimer()
    timer.start()
    while not until() and timer.elapsed() < timeout_ms:  # type: ignore[operator]
        QCoreApplication.processEvents()


def test_token_file_is_private(tmp_path: Path) -> None:
    token = create_token(tmp_path)
    assert read_token(tmp_path) == token
    assert stat.S_IMODE((tmp_path / "ws-token").stat().st_mode) == 0o600
    assert check_hello(hello(token), token)
    assert not check_hello(hello("faux"), token)
    assert not check_hello("pas du json", token)


def test_parse_event_filters_unknown() -> None:
    assert parse_event('{"type": "state", "state": "alert"}') == {"type": "state", "state": "alert"}
    assert parse_event('{"type": "state", "state": "danse"}') is None
    assert parse_event('{"type": "run", "cmd": "rm -rf /"}') is None


def test_window_reacts_to_events(app: QApplication) -> None:
    window = CompanionWindow(load_companions())
    window.place_default()
    window.handle_event({"type": "focus", "companion": "docker", "file": "/p/Dockerfile"})
    assert window.current.name == "docker"
    window.handle_event({"type": "state", "state": "alert"})
    assert window.sprite.state == "alert"
    window.handle_event(
        {
            "type": "diagnostics",
            "items": [
                {"line": 1, "severity": "warning", "code": "DL3006", "message": "Tag <latest>"},
                {"line": 2, "severity": "error", "code": "DL4000", "message": "MAINTAINER obsolète"},
            ],
        }
    )
    text = window.bubble.label.text()
    assert "Ligne 2" in text and "DL4000" in text
    assert "&lt;latest&gt;" not in text  # seule la première erreur est montrée
    window.handle_event({"type": "diagnostics", "items": []})
    assert window.bubble.isHidden()
    window.handle_event({"type": "suggestion", "text": "<script>x</script>"})
    assert "&lt;script&gt;" in window.bubble.label.text()  # texte échappé, jamais interprété


def test_bridge_auth_and_dispatch(app: QApplication, tmp_path: Path) -> None:
    """Un vrai client LSP (asyncio, websockets) parle à la vraie fenêtre (Qt)."""
    token = create_token(tmp_path)
    server = BridgeServer(token, port=0)
    received: list[dict[str, object]] = []
    server.event_received.connect(received.append)
    counts: list[int] = []
    server.clients_changed.connect(counts.append)
    result: dict[str, object] = {}

    async def scenario() -> None:
        from websockets.asyncio.client import connect

        # 1. Sans jeton : refusé, rien n'est transmis.
        async with connect(f"ws://127.0.0.1:{server.port}", proxy=None) as ws:
            await ws.send(json.dumps({"type": "hello", "token": "faux"}))
            await ws.send(json.dumps({"type": "state", "state": "alert"}))
            try:
                await asyncio.wait_for(ws.recv(), 2)
            except Exception as exc:
                result["refused"] = type(exc).__name__
        # 2. Avec le client officiel : accepté.
        client = BridgeClient(tmp_path, port=server.port)
        client.start()
        await asyncio.wait_for(client.connected.wait(), 5)
        client.emit({"type": "focus", "companion": "kubernetes", "file": "/p/k.yaml"})
        client.emit({"type": "state", "state": "think"})
        await asyncio.sleep(0.5)

    thread = threading.Thread(target=lambda: asyncio.run(scenario()))
    thread.start()
    pump(lambda: not thread.is_alive(), 15000)
    thread.join(1)
    pump(lambda: len(received) >= 2, 2000)
    assert "refused" in result
    assert received == [
        {"type": "focus", "companion": "kubernetes", "file": "/p/k.yaml"},
        {"type": "state", "state": "think"},
    ]
    pump(lambda: counts == [1, 0], 2000)
    assert counts == [1, 0]  # seul le client authentifié a compté, puis s'est déconnecté
    assert server.active is None
    server.close()


def test_window_plays_behavior_animations(app: QApplication) -> None:
    now = [0.0]
    window = CompanionWindow(load_companions(), clock=lambda: now[0])
    sprite = window.sprite
    assert sprite.sheets is not None  # bandes PNG du chat chargées
    frame = sprite.sheets.frame("mange", 1.0)
    assert frame is not None and frame.width() == 256 and not frame.isNull()
    clicks: list[bool] = []
    window.sprite_clicked.connect(lambda: clicks.append(True))

    sprite.clicked.emit()  # clic simple : il est content et la question s'ouvre
    assert sprite.anim == "content" and clicks == [True]

    window.handle_event({"type": "diagnostics", "items": []})
    now[0] = 5
    window.handle_event({"type": "diagnostics", "items": [{"severity": "error", "line": 1}]})
    window.tick_behavior()
    assert sprite.anim == "surpris"  # nouvelle erreur

    for k in range(5):  # on le déplace sans arrêt : il boude
        window.behavior.drag_start(10 + k, 0, 0)
        window.behavior.drag_move(10.5 + k, 50, 0)
        window.behavior.drag_end(10.5 + k, moved=True)
    now[0] = 15
    window.tick_behavior()
    assert sprite.anim == "fache"
    sprite.clicked.emit()
    assert clicks == [True]  # clic ignoré
    assert "boude" in window.bubble.label.text()

    window.handle_event({"type": "focus", "companion": "kubernetes", "file": "/p/k.yaml"})
    assert window.sprite.sheets is not None and "dort" in window.sprite.sheets.clips
    window.grab()  # le rendu d'une image ne plante pas
