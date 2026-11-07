"""Étape 7 : cliquer sur l'expert pour lui poser une question."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from lsprotocol import types as lsp
from pygls.lsp.client import LanguageClient
from PySide6.QtCore import QCoreApplication, QElapsedTimer
from PySide6.QtWidgets import QApplication

from experts.core.app import build_runtime
from experts.core.bridge import create_token
from experts.core.companions import load_companions
from experts.core.config import Settings
from experts.core.fixer import TextEdit
from experts.core.llm import Message, ToolCall
from experts.lsp.ask import (
    AskService,
    EditorContext,
    build_context,
    enclosing_python_symbol,
    insertion_edit,
    source_label,
)
from experts.ui.bridge_server import BridgeServer
from experts.ui.controller import Controller
from experts.ui.question import last_code_block
from experts.ui.window import CompanionWindow

CODE = '''import subprocess


def run_ruff(path: str) -> list[str]:
    """Lance ruff."""
    proc = subprocess.run(["ruff", path], capture_output=True)
    return proc.stdout.splitlines()
'''
ANSWER = (
    'Passe `timeout=` à subprocess.run.\n```python\nproc = subprocess.run(["ruff", path], timeout=10)\n```'
)


@pytest.fixture(scope="module")
def app() -> Iterator[QApplication]:
    instance = QApplication.instance() or QApplication([])
    assert isinstance(instance, QApplication)
    yield instance


def pump(until: Any, timeout_ms: int = 5000) -> None:
    timer = QElapsedTimer()
    timer.start()
    while not until() and timer.elapsed() < timeout_ms:
        QCoreApplication.processEvents()


# --- Contexte joint automatiquement -------------------------------------------------------------


def test_context_uses_enclosing_function() -> None:
    assert enclosing_python_symbol(CODE, 5) == ("run_ruff", 3, 6)
    ctx = build_context("file:///p/validators.py", Path("/p/validators.py"), CODE, (5, 5))
    assert ctx.label == "validators.py › run_ruff · l. 4-7"  # noqa: RUF001
    assert ctx.snippet().startswith("   4 | def run_ruff")


def test_context_without_cursor_or_outside_python() -> None:
    ctx = build_context("u", Path("/p/Dockerfile"), "FROM x\n" * 100, None)
    assert (ctx.start, ctx.end, ctx.symbol) == (0, 29, None)
    ctx = build_context("u", Path("/p/Dockerfile"), "FROM x\n" * 100, (50, 50))
    assert (ctx.start, ctx.end) == (35, 65)


def test_insertion_edit() -> None:
    ctx = EditorContext("u", Path("a.py"), "a\nb\nc\n", 0, 2)
    assert insertion_edit(ctx, "X", (1, 2)) == TextEdit(1, 3, "X\n")  # remplace la sélection
    assert insertion_edit(ctx, "X", (1, 1)) == TextEdit(2, 2, "X\n")  # sous le curseur
    assert insertion_edit(ctx, "X", None) == TextEdit(3, 3, "X\n")  # en fin de fichier


def test_source_labels() -> None:
    assert (
        source_label(ToolCall("search_docs", {"collection": "python", "query": "x"})) == "docs python (RAG)"
    )
    assert source_label(ToolCall("web_search", {"query": "ruff timeout"})) == "SearXNG · ruff timeout"
    assert source_label(ToolCall("list_files", {})) is None


# --- Service côté LSP ---------------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, *replies: Message) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: object = None) -> Message:
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def close(self) -> None:
        pass


def make_service(tmp_path: Path, llm: ScriptedLLM) -> tuple[AskService, list[dict[str, Any]], list[Any]]:
    (tmp_path / "validators.py").write_text(CODE)
    settings = replace(Settings.from_env(), workspace=tmp_path, data_dir=tmp_path / "data")
    service: AskService | None = None
    runtime = build_runtime(
        settings, confirm=lambda argv: service.confirm(argv) if service else False, llm=llm
    )
    sent: list[dict[str, Any]] = []
    applied: list[Any] = []
    ctx = build_context("file:///v", tmp_path / "validators.py", CODE, (5, 5))

    async def apply_edit(uri: str, edit: TextEdit) -> bool:
        applied.append((uri, edit))
        return True

    service = AskService(runtime, sent.append, lambda: ctx, lambda uri: (5, 5), apply_edit)
    return service, sent, applied


def test_ask_attaches_context_and_sources(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        Message(
            role="assistant",
            tool_calls=[ToolCall("search_docs", {"query": "timeout", "collection": "python"})],
        ),
        Message(role="assistant", content=ANSWER),
    )
    service, sent, _ = make_service(tmp_path, llm)
    asyncio.run(service.handle({"type": "ask", "id": "q1", "question": "Comment ajouter un timeout ?"}))
    answer = next(e for e in sent if e["type"] == "answer")
    assert answer["id"] == "q1" and answer["companion"] == "python"
    assert answer["context"] == "validators.py › run_ruff · l. 4-7"  # noqa: RUF001
    assert answer["sources"] == ["docs python (RAG)"]
    prompt = llm.seen[0][-1].content
    assert "def run_ruff" in prompt and "Comment ajouter un timeout ?" in prompt
    assert [e["state"] for e in sent if e["type"] == "state"] == ["think", "idle"]


def test_command_needs_window_confirmation(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        Message(role="assistant", tool_calls=[ToolCall("run_command", {"command": "touch pirate.txt"})]),
        Message(role="assistant", content="D'accord, je ne lance rien."),
    )
    service, sent, _ = make_service(tmp_path, llm)

    async def scenario() -> None:
        task = asyncio.ensure_future(
            service.handle({"type": "ask", "id": "q", "question": "crée un fichier"})
        )
        for _ in range(200):
            confirm = next((e for e in sent if e["type"] == "confirm"), None)
            if confirm:
                break
            await asyncio.sleep(0.01)
        assert confirm is not None and confirm["command"] == "touch pirate.txt"
        await service.handle({"type": "confirm_reply", "id": confirm["id"], "ok": "oui"})  # pas un vrai True
        await task

    asyncio.run(scenario())
    assert not (tmp_path / "pirate.txt").exists()
    assert "Refusé" in llm.seen[1][-1].content


def test_insert_shows_diff_then_applies(tmp_path: Path) -> None:
    service, sent, applied = make_service(tmp_path, ScriptedLLM())
    asyncio.run(service.handle({"type": "insert_preview", "id": "p1", "code": "    print(proc)"}))
    diff = next(e for e in sent if e["type"] == "diff")
    assert "+    print(proc)" in diff["diff"] and applied == []  # rien d'écrit à ce stade
    asyncio.run(service.handle({"type": "insert_apply", "id": "p1"}))
    assert applied == [("file:///v", TextEdit(6, 6, "    print(proc)\n"))]
    asyncio.run(service.handle({"type": "insert_apply", "id": "p1"}))  # un aperçu ne sert qu'une fois
    assert len(applied) == 1


# --- Fenêtre ------------------------------------------------------------------------------------


class FakeBridge:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> bool:
        self.sent.append(message)
        return True


def test_click_opens_panel_and_full_flow(app: QApplication) -> None:
    window = CompanionWindow(load_companions())
    window.place_default()
    bridge = FakeBridge()
    ctrl = Controller(window, bridge)
    ctrl.confirm_diff = lambda dialog: True

    window.sprite.clicked.emit()
    assert ctrl.panel.isVisible() and window.sprite.selected
    ctrl.panel.input.setText("Comment ajouter un timeout ?")
    ctrl.panel.input.returnPressed.emit()
    assert bridge.sent[-1]["type"] == "ask" and not ctrl.panel.input.isEnabled()

    ctrl.handle_event(
        {
            "type": "answer",
            "id": "x",
            "companion": "python",
            "context": "validators.py › run_ruff · l. 14-29",  # noqa: RUF001
            "text": ANSWER,
            "sources": ["docs python (RAG)", "SearXNG · ruff"],
        }
    )
    assert "run_ruff" in ctrl.panel.chip.text()
    assert ctrl.panel.insert_btn.isEnabled() and ctrl.panel.input.isEnabled()
    ctrl.panel.insert_btn.click()
    assert bridge.sent[-1] == {
        "type": "insert_preview",
        "id": bridge.sent[-1]["id"],
        "code": last_code_block(ANSWER),
    }

    ctrl.handle_event({"type": "diff", "id": "p", "file": "validators.py", "diff": "+x\n"})
    assert bridge.sent[-1] == {"type": "insert_apply", "id": "p"}

    ctrl.handle_event({"type": "confirm", "id": "c1", "command": "rm -rf build"})
    assert ctrl.panel.confirm_box.isVisible()
    ctrl.panel.deny.click()
    assert bridge.sent[-1] == {"type": "confirm_reply", "id": "c1", "ok": False}

    window.sprite.clicked.emit()
    assert not ctrl.panel.isVisible() and not window.sprite.selected


# --- Bout en bout : vraie fenêtre (Qt) + vrai serveur LSP + faux Ollama -------------------------


@pytest.fixture
def fake_ollama() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"message": {"role": "assistant", "content": ANSWER}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_end_to_end_question(app: QApplication, tmp_path: Path, fake_ollama: str) -> None:
    data = tmp_path / "data"
    bridge = BridgeServer(create_token(data), port=0)
    events: list[dict[str, Any]] = []
    bridge.event_received.connect(events.append)
    source = tmp_path / "validators.py"
    source.write_text(CODE)
    done = threading.Event()

    async def editor() -> None:
        client = LanguageClient("test", "1")
        env = {
            **os.environ,
            "EXPERTS_OLLAMA_HOST": fake_ollama,
            "EXPERTS_DATA_DIR": str(data),
            "EXPERTS_UI_PORT": str(bridge.port),
        }
        await client.start_io(sys.executable, "-m", "experts.lsp", env=env)
        await client.initialize_async(
            lsp.InitializeParams(capabilities=lsp.ClientCapabilities(), root_uri=tmp_path.as_uri())
        )
        client.initialized(lsp.InitializedParams())
        await asyncio.sleep(1.5)  # le temps que le serveur LSP rejoigne la fenêtre
        client.text_document_did_open(
            lsp.DidOpenTextDocumentParams(
                lsp.TextDocumentItem(uri=source.as_uri(), language_id="python", version=1, text=CODE)
            )
        )
        while not done.is_set():
            await asyncio.sleep(0.05)
        await client.shutdown_async(None)
        client.exit(None)
        await client.stop()  # type: ignore[no-untyped-call]

    thread = threading.Thread(target=lambda: asyncio.run(editor()))
    thread.start()
    try:
        pump(lambda: any(e["type"] == "focus" for e in events), 20000)
        assert bridge.send({"type": "ask", "id": "q1", "question": "Comment ajouter un timeout ?"})
        pump(lambda: any(e["type"] == "answer" for e in events), 20000)
        answer = next(e for e in events if e["type"] == "answer")
        assert answer["id"] == "q1" and answer["companion"] == "python" and "timeout=" in answer["text"]
        assert answer["context"].startswith("validators.py")
    finally:
        done.set()
        pump(lambda: not thread.is_alive(), 10000)
        thread.join(2)
        bridge.close()
