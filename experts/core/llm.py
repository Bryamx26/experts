"""Client minimal pour l'API /api/chat d'Ollama, avec appel d'outils.

Choix : httpx directement plutôt que la bibliothèque `ollama`, pour garder la
main sur les délais, les erreurs et pouvoir tester avec un faux transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(RuntimeError):
    """Ollama injoignable ou réponse inattendue."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None

    def to_api(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls
            ]
        if self.tool_name:
            data["tool_name"] = self.tool_name
        return data


def _parse_message(payload: dict[str, Any]) -> Message:
    raw = payload.get("message")
    if not isinstance(raw, dict):
        raise LLMError(f"Réponse Ollama sans champ 'message' : {payload!r}")
    calls: list[ToolCall] = []
    for item in raw.get("tool_calls") or []:
        fn = item.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(fn.get("name"), str) and isinstance(args, dict):
            calls.append(ToolCall(name=fn["name"], arguments=args))
    return Message(role="assistant", content=raw.get("content") or "", tool_calls=calls)


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout: float = 120.0,
        think: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        # think=False : Qwen3.x raisonne moins longtemps, la latence compte plus ici.
        self.think = think
        self._http = httpx.Client(base_url=host, timeout=timeout, transport=transport)

    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Message:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "stream": False,
            "think": self.think,
        }
        if tools:
            body["tools"] = tools
        try:
            response = self._http.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(f"Ollama injoignable sur {self._http.base_url}. Lance `ollama serve`.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama a répondu {exc.response.status_code} : {exc.response.text}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama n'a pas répondu à temps.") from exc
        return _parse_message(response.json())

    def close(self) -> None:
        self._http.close()
