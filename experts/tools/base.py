"""Socle commun des outils : description JSON Schema + registre.

Choix : une dataclass plutôt qu'une classe abstraite par outil. Chaque outil
est une simple fonction typée, plus facile à tester et à déclarer en YAML à
l'étape 3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_OUTPUT_CHARS = 8_000


class ToolError(Exception):
    """Erreur prévue d'un outil, renvoyée au modèle sous forme de texte."""


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Coupe les sorties longues : un petit modèle a un contexte limité."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[… {len(text) - limit} caractères coupés]"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Outil déjà enregistré : {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str] | tuple[str, ...]) -> ToolRegistry:
        """Registre limité aux outils autorisés pour un compagnon (ignore les noms inconnus)."""
        return ToolRegistry([self._tools[n] for n in names if n in self._tools])

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Exécute un outil. Ne lève jamais : le modèle reçoit l'erreur en texte."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Erreur : outil inconnu '{name}'. Outils disponibles : {', '.join(self._tools)}."
        try:
            return truncate(tool.func(**arguments))
        except ToolError as exc:
            return f"Erreur : {exc}"
        except TypeError as exc:
            return f"Erreur : arguments invalides pour '{name}' ({exc})."
