"""Tests de l'étape 1, sans Ollama ni SearXNG (faux transports httpx)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from experts.core.agent import Agent
from experts.core.llm import LLMError, Message, OllamaClient
from experts.tools.base import ToolRegistry
from experts.tools.files import make_list_files, make_read_file
from experts.tools.shell import make_run_command
from experts.tools.web_search import make_web_search


def ollama_transport(replies: list[dict[str, Any]], seen: list[dict[str, Any]]) -> httpx.MockTransport:
    """Renvoie les réponses dans l'ordre et garde les requêtes reçues."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": queue.pop(0)})

    return httpx.MockTransport(handler)


def test_agent_runs_tool_then_answers(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("import os\nprint('ok')\n")
    seen: list[dict[str, Any]] = []
    transport = ollama_transport(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "app.py"}}}],
            },
            {"role": "assistant", "content": "Ligne 1 : import os inutilisé."},
        ],
        seen,
    )
    llm = OllamaClient("http://ollama.test", "qwen3.5:4b", transport=transport)
    agent = Agent(llm, ToolRegistry([make_read_file(tmp_path)]))

    assert agent.ask("Qu'est-ce qui cloche dans app.py ?") == "Ligne 1 : import os inutilisé."
    assert seen[0]["tools"][0]["function"]["name"] == "read_file"
    tool_msg = seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_name"] == "read_file"
    assert "    1 | import os" in tool_msg["content"]


def test_agent_stops_after_max_rounds(tmp_path: Path) -> None:
    looping = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}],
    }
    seen: list[dict[str, Any]] = []
    transport = ollama_transport([looping, looping, {"role": "assistant", "content": "fini"}], seen)
    llm = OllamaClient("http://ollama.test", "m", transport=transport)
    agent = Agent(llm, ToolRegistry([make_list_files(tmp_path)]), max_tool_rounds=2)

    assert agent.ask("boucle") == "fini"
    assert "tools" not in seen[-1]  # dernier tour forcé sans outils


def test_ollama_unreachable_gives_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm = OllamaClient("http://ollama.test", "m", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="ollama serve"):
        llm.chat([Message(role="user", content="salut")])


def test_read_file_refuses_path_traversal(tmp_path: Path) -> None:
    registry = ToolRegistry([make_read_file(tmp_path / "ws")])
    (tmp_path / "ws").mkdir()
    (tmp_path / "secret.txt").write_text("motdepasse")
    result = registry.call("read_file", {"path": "../secret.txt"})
    assert "hors de l'espace de travail" in result
    assert "motdepasse" not in result


def test_read_file_refuses_symlink_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.txt").write_text("motdepasse")
    (ws / "lien").symlink_to(tmp_path / "secret.txt")
    assert "hors de l'espace de travail" in ToolRegistry([make_read_file(ws)]).call(
        "read_file", {"path": "lien"}
    )


def test_run_command_refused_does_not_execute(tmp_path: Path) -> None:
    asked: list[list[str]] = []

    def deny(argv: list[str]) -> bool:
        asked.append(argv)
        return False

    registry = ToolRegistry([make_run_command(tmp_path, deny)])
    result = registry.call("run_command", {"command": "touch cree.txt"})
    assert asked == [["touch", "cree.txt"]]
    assert "Refusé" in result
    assert not (tmp_path / "cree.txt").exists()


def test_run_command_has_no_shell_injection(tmp_path: Path) -> None:
    registry = ToolRegistry([make_run_command(tmp_path, lambda argv: True)])
    result = registry.call("run_command", {"command": "echo a; touch pirate.txt"})
    assert "a; touch pirate.txt" in result  # passé tel quel à echo, pas interprété
    assert not (tmp_path / "pirate.txt").exists()


def test_unknown_tool_and_bad_arguments(tmp_path: Path) -> None:
    registry = ToolRegistry([make_read_file(tmp_path)])
    assert "outil inconnu" in registry.call("rm_rf", {})
    assert "arguments invalides" in registry.call("read_file", {"chemin": "x"})


def test_web_search_formats_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Ruff", "url": "https://docs.astral.sh/ruff/", "content": "Linter Python."}
                ]
            },
        )

    tool = make_web_search("http://searx.test", transport=httpx.MockTransport(handler))
    out = ToolRegistry([tool]).call("web_search", {"query": "ruff"})
    assert "[1] Ruff" in out and "https://docs.astral.sh/ruff/" in out
