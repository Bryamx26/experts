"""Étape 4 : correcteur et serveur LSP de bout en bout (vrai processus, faux Ollama)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from lsprotocol import types as lsp
from pygls.lsp.client import LanguageClient

from experts.core.companions import load_companions
from experts.core.fixer import extract_code, is_accepted, minimal_edit, propose_fix
from experts.core.llm import Message
from experts.tools.validators import HADOLINT, Diagnostic, validate_content

needs_hadolint = pytest.mark.skipif(shutil.which("hadolint") is None, reason="hadolint absent")
FIXED_ANSWER = "L'image doit être épinglée.\n```dockerfile\nFROM ubuntu:24.04\n```\n"


class ScriptedLLM:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.seen: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: object = None) -> Message:
        self.seen.append(list(messages))
        return Message(role="assistant", content=self.answers.pop(0))

    def close(self) -> None:
        pass


def diag(code: str, line: int = 1, severity: str = "warning") -> Diagnostic:
    return Diagnostic("Dockerfile", line, 1, severity, code, "msg", "hadolint")  # type: ignore[arg-type]


def test_extract_code() -> None:
    assert extract_code(FIXED_ANSWER) == ("L'image doit être épinglée.", "FROM ubuntu:24.04\n")
    assert extract_code("pas de code") is None


def test_minimal_edit_touches_only_changed_lines() -> None:
    edit = minimal_edit("a\nb\nc\n", "a\nB\nc\n")
    assert edit is not None and (edit.start_line, edit.end_line, edit.text) == (1, 2, "B\n")
    assert minimal_edit("a\n", "a\n") is None


def test_is_accepted() -> None:
    target = diag("DL3006")
    assert is_accepted(target, [target], [])
    assert not is_accepted(target, [target], [diag("DL3006", line=2)])
    assert not is_accepted(diag("E1", severity="error"), [], [diag("E2", severity="error")])


@needs_hadolint
def test_propose_fix_retries_with_validator_feedback() -> None:
    spec = load_companions()["docker"]
    before = validate_content("Dockerfile", "FROM ubuntu\n", [HADOLINT])
    target = next(d for d in before if d.code == "DL3006")
    llm = ScriptedLLM("Voilà.\n```\nFROM ubuntu\n```", FIXED_ANSWER)
    fix = propose_fix(llm, spec, "Dockerfile", "FROM ubuntu\n", target, [HADOLINT], before)
    assert fix is not None and fix.content == "FROM ubuntu:24.04\n"
    assert "refusent encore" in llm.seen[1][-1].content  # 2e essai avec la sortie de hadolint


@needs_hadolint
def test_propose_fix_gives_up() -> None:
    spec = load_companions()["docker"]
    before = validate_content("Dockerfile", "FROM ubuntu\n", [HADOLINT])
    target = next(d for d in before if d.code == "DL3006")
    llm = ScriptedLLM("```\nFROM ubuntu\n```", "```\nFROM ubuntu\n```")
    assert propose_fix(llm, spec, "Dockerfile", "FROM ubuntu\n", target, [HADOLINT], before) is None


# --- Bout en bout ---------------------------------------------------------------------------


@pytest.fixture
def fake_ollama() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"message": {"role": "assistant", "content": FIXED_ANSWER}}).encode()
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


@needs_hadolint
def test_lsp_end_to_end(tmp_path: Path, fake_ollama: str) -> None:
    asyncio.run(_scenario(tmp_path, fake_ollama))


async def _scenario(tmp_path: Path, ollama: str) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM ubuntu\n")
    uri = dockerfile.as_uri()
    received: asyncio.Queue[lsp.PublishDiagnosticsParams] = asyncio.Queue()

    client = LanguageClient("test", "1")

    @client.feature(lsp.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)  # type: ignore[untyped-decorator]
    def on_diags(params: lsp.PublishDiagnosticsParams) -> None:
        received.put_nowait(params)

    env = {**os.environ, "EXPERTS_OLLAMA_HOST": ollama, "EXPERTS_DATA_DIR": str(tmp_path / "data")}
    await client.start_io(sys.executable, "-m", "experts.lsp", env=env)
    try:
        await client.initialize_async(
            lsp.InitializeParams(capabilities=lsp.ClientCapabilities(), root_uri=tmp_path.as_uri())
        )
        client.initialized(lsp.InitializedParams())
        client.text_document_did_open(
            lsp.DidOpenTextDocumentParams(
                lsp.TextDocumentItem(uri=uri, language_id="dockerfile", version=1, text="FROM ubuntu\n")
            )
        )
        params = await asyncio.wait_for(received.get(), 20)
        dl3006 = next(d for d in params.diagnostics if d.code == "DL3006")
        assert dl3006.range.start.line == 0 and dl3006.source == "experts/hadolint"

        actions = await client.text_document_code_action_async(
            lsp.CodeActionParams(
                text_document=lsp.TextDocumentIdentifier(uri),
                range=dl3006.range,
                context=lsp.CodeActionContext(diagnostics=[dl3006]),
            )
        )
        assert actions and isinstance(actions[0], lsp.CodeAction)
        assert actions[0].title == "Docker : corriger « DL3006 »"
        resolved = await client.protocol.send_request_async(lsp.CODE_ACTION_RESOLVE, actions[0])
        assert resolved.edit is not None and resolved.edit.changes is not None
        assert resolved.edit.changes[uri][0].new_text == "FROM ubuntu:24.04\n"
    finally:
        await client.shutdown_async(None)
        client.exit(None)
        await client.stop()  # type: ignore[no-untyped-call]
