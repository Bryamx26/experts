"""Étape 6 : complétion FIM (texte grisé)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from lsprotocol import types as lsp
from pygls.lsp.client import LanguageClient

from experts.core.fim import FimClient, clean_completion


def test_clean_completion() -> None:
    assert clean_completion("return a + b\n", "") == "return a + b"
    assert clean_completion("   \n", "") == ""
    assert clean_completion("x = 1\n" * 20, "").count("\n") == 7  # 8 lignes au plus
    # Le modèle répète la ligne qui suit le curseur : on coupe avant.
    assert clean_completion("total += x\n    return total", "\n    return total\n") == "total += x"


def test_fim_client_sends_prefix_and_suffix() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "a + b"})

    client = FimClient("http://ollama.test", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.complete("x" * 5000 + "return ", "\n")) == "a + b"
    body = seen[0]
    assert body["model"] == "qwen2.5-coder:1.5b" and body["suffix"] == "\n"
    assert len(body["prompt"]) == 3000 and body["prompt"].endswith("return ")


def test_fim_client_is_silent_when_ollama_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = FimClient("http://ollama.test", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.complete("a", "b")) == ""


@pytest.fixture
def fake_ollama() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"response": "a + b"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", seen
    httpd.shutdown()


def test_lsp_inline_completion(tmp_path: Path, fake_ollama: tuple[str, list[dict[str, Any]]]) -> None:
    url, seen = fake_ollama
    asyncio.run(_scenario(tmp_path, url))
    assert seen[-1]["prompt"].endswith("    return ") and seen[-1]["suffix"] == "\n"


async def _scenario(tmp_path: Path, ollama: str) -> None:
    source = "def add(a: int, b: int) -> int:\n    return \n"
    path = tmp_path / "calc.py"
    path.write_text(source)
    client = LanguageClient("test", "1")
    env = {**os.environ, "EXPERTS_OLLAMA_HOST": ollama, "EXPERTS_DATA_DIR": str(tmp_path / "data")}
    await client.start_io(sys.executable, "-m", "experts.lsp", env=env)
    try:
        init = await client.initialize_async(
            lsp.InitializeParams(capabilities=lsp.ClientCapabilities(), root_uri=tmp_path.as_uri())
        )
        assert init.capabilities.inline_completion_provider
        client.initialized(lsp.InitializedParams())
        uri = path.as_uri()
        client.text_document_did_open(
            lsp.DidOpenTextDocumentParams(
                lsp.TextDocumentItem(uri=uri, language_id="python", version=1, text=source)
            )
        )
        result = await client.protocol.send_request_async(
            lsp.TEXT_DOCUMENT_INLINE_COMPLETION,
            lsp.InlineCompletionParams(
                context=lsp.InlineCompletionContext(trigger_kind=lsp.InlineCompletionTriggerKind.Automatic),
                text_document=lsp.TextDocumentIdentifier(uri),
                position=lsp.Position(1, 11),
            ),
        )
        assert result.items[0].insert_text == "a + b"
        assert result.items[0].range.start == lsp.Position(1, 11)
    finally:
        await client.shutdown_async(None)
        client.exit(None)
        await client.stop()  # type: ignore[no-untyped-call]
