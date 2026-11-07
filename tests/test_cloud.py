"""Tests du mode cloud, sans réseau : faux client imitant le SDK anthropic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experts.core.agent import Agent
from experts.core.cloud import AnthropicClient, to_anthropic_messages
from experts.core.llm import LLMError, Message, ToolCall
from experts.tools.base import ToolRegistry
from experts.tools.files import make_read_file


def block(type_: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **fields)


class FakeAnthropic:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params: Any) -> SimpleNamespace:
        self.calls.append(params)
        return self._responses.pop(0)

    def close(self) -> None:
        pass


def response(*content: SimpleNamespace, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=list(content), stop_reason=stop)


def test_cloud_tool_loop(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n")
    thinking = block("thinking", thinking="", signature="sig")
    fake = FakeAnthropic(
        [
            response(
                thinking,
                block("tool_use", id="tu_1", name="read_file", input={"path": "app.py"}),
                stop="tool_use",
            ),
            response(block("text", text="Ligne 1 : x vaut 1.")),
        ]
    )
    agent = Agent(AnthropicClient(client=fake), ToolRegistry([make_read_file(tmp_path)]))

    assert agent.ask("Que fait app.py ?") == "Ligne 1 : x vaut 1."
    first = fake.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default"
    assert first["tools"][0]["input_schema"]["required"] == ["path"]
    assert "compagnon Python" in first["system"]
    second = fake.calls[1]["messages"]
    assert second[1]["content"][0] is thinking  # bloc de réflexion rejoué tel quel
    assert second[2]["content"][0]["tool_use_id"] == "tu_1"


def test_parallel_tool_results_grouped_in_one_user_message() -> None:
    history = [
        Message(role="system", content="S"),
        Message(role="user", content="Q"),
        Message(
            role="assistant",
            tool_calls=[ToolCall("a", {}, id="1"), ToolCall("b", {}, id="2")],
        ),
        Message(role="tool", content="ra", tool_call_id="1"),
        Message(role="tool", content="rb", tool_call_id="2"),
    ]
    system, msgs = to_anthropic_messages(history)
    assert system == "S"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["1", "2"]


def test_fallback_boundary_drops_internal_blocks() -> None:
    raw = [
        block("thinking", thinking=""),
        block("text", text="début"),
        block("fallback"),
        block("thinking", thinking=""),
        block("text", text="suite"),
    ]
    _, msgs = to_anthropic_messages([Message(role="user", content="Q"), Message(role="assistant", raw=raw)])
    kept = [b.type for b in msgs[1]["content"]]
    assert kept == ["text", "fallback", "thinking", "text"]


def test_refusal_raises() -> None:
    fake = FakeAnthropic([response(stop="refusal")])
    with pytest.raises(LLMError, match="refusé"):
        AnthropicClient(client=fake).chat([Message(role="user", content="Q")])


def test_secret_files_are_never_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=abc")
    (tmp_path / "server.key").write_text("-----BEGIN")
    registry = ToolRegistry([make_read_file(tmp_path)])
    for name in (".env", "server.key"):
        out = registry.call("read_file", {"path": name})
        assert "secrets" in out and "abc" not in out
