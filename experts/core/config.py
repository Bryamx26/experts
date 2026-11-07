"""Configuration lue depuis l'environnement.

Choix : des variables d'environnement plutôt qu'un fichier, pour l'étape 1.
Les valeurs par défaut pointent toutes vers localhost : rien ne sort de la machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3.5:4b"
    searxng_url: str = "http://127.0.0.1:8888"
    workspace: Path = field(default_factory=Path.cwd)
    request_timeout: float = 120.0
    max_tool_rounds: int = 6

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_host=os.environ.get("COMPANION_OLLAMA_HOST", cls.ollama_host),
            chat_model=os.environ.get("COMPANION_CHAT_MODEL", cls.chat_model),
            searxng_url=os.environ.get("COMPANION_SEARXNG_URL", cls.searxng_url),
            workspace=Path(os.environ.get("COMPANION_WORKSPACE", Path.cwd())).resolve(),
            request_timeout=_env_float("COMPANION_TIMEOUT", cls.request_timeout),
        )
