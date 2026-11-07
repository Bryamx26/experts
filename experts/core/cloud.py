"""Client optionnel pour un modèle payant (API Claude d'Anthropic).

Désactivé par défaut : le projet reste « tout local ». À activer explicitement
avec `--cloud` ou EXPERTS_BACKEND=anthropic, quand le petit modèle local ne
suffit pas (explication difficile, gros refactor).

Choix techniques :
- SDK officiel `anthropic`, installé seulement avec l'extra `[cloud]` ;
  l'import est différé pour que le mode local n'en dépende pas.
- Même interface `chat()` qu'OllamaClient : l'agent et les outils ne changent pas.
- Le contenu brut de chaque réponse est renvoyé tel quel au tour suivant, car
  les blocs de réflexion de Claude doivent être rejoués sans modification.
- `fallbacks: "default"` : si les filtres de sécurité refusent une demande
  (faux positif possible sur du code réseau/sécurité), l'API la relance sur
  un autre modèle au lieu de renvoyer un refus.
"""

from __future__ import annotations

from typing import Any

from experts.core.llm import LLMError, Message, ToolCall

DEFAULT_CLOUD_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_DROP_BEFORE_FALLBACK = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}


def _anthropic_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Convertit un schéma d'outil au format Ollama/OpenAI vers le format Anthropic."""
    fn = schema["function"]
    return {"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]}


def _replayable(content: list[Any]) -> list[Any]:
    """Après un repli en cours de réponse, retire les blocs internes d'avant le dernier repli."""
    last = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
    return [b for i, b in enumerate(content) if i > last or b.type not in _DROP_BEFORE_FALLBACK]


def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Sépare le prompt système et regroupe les résultats d'outils consécutifs.

    L'API attend tous les tool_result d'un tour dans un seul message `user`.
    """
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            if m.raw is not None:
                out.append({"role": "assistant", "content": _replayable(m.raw)})
            else:  # réponse venue d'un autre fournisseur : on reconstruit
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m.content})
    return system, out


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_CLOUD_MODEL,
        *,
        effort: str = "medium",
        max_tokens: int = 16_000,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("Mode cloud : installe l'extra avec `pip install -e '.[cloud]'`.") from exc
            # La clé vient de ANTHROPIC_API_KEY (ou `ant auth login`), jamais du code.
            client = anthropic.Anthropic()
        self.model = model
        self._client = client
        self._effort = effort
        self._max_tokens = max_tokens

    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Message:
        system, api_messages = to_anthropic_messages(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": api_messages,
            "output_config": {"effort": self._effort},
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            params["tools"] = [_anthropic_tool(t) for t in tools]
        try:
            response = self._client.beta.messages.create(**params)
        except Exception as exc:  # erreurs typées du SDK, importé paresseusement
            raise LLMError(f"API Claude : {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Le modèle cloud a refusé cette demande ; reformule-la.")
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(name=b.name, arguments=dict(b.input), id=b.id)
            for b in response.content
            if b.type == "tool_use"
        ]
        return Message(role="assistant", content=text, tool_calls=calls, raw=list(response.content))

    def close(self) -> None:
        self._client.close()
