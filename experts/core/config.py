"""Configuration lue depuis l'environnement.

Choix : des variables d'environnement plutôt qu'un fichier, pour l'étape 1.
Les valeurs par défaut pointent toutes vers localhost : rien ne sort de la machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["ollama", "anthropic"]


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
    # "anthropic" = modèle payant dans le cloud, à activer explicitement.
    backend: Backend = "ollama"
    cloud_model: str = "claude-opus-5-5"
    embed_model: str = "nomic-embed-text"
    fim_model: str = "qwen2.5-coder:1.5b"
    ui_port: int = 8765
    data_dir: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "experts")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_host=os.environ.get("EXPERTS_OLLAMA_HOST", cls.ollama_host),
            chat_model=os.environ.get("EXPERTS_CHAT_MODEL", cls.chat_model),
            searxng_url=os.environ.get("EXPERTS_SEARXNG_URL", cls.searxng_url),
            workspace=Path(os.environ.get("EXPERTS_WORKSPACE", Path.cwd())).resolve(),
            request_timeout=_env_float("EXPERTS_TIMEOUT", cls.request_timeout),
            backend="anthropic" if os.environ.get("EXPERTS_BACKEND") == "anthropic" else "ollama",
            cloud_model=os.environ.get("EXPERTS_CLOUD_MODEL", cls.cloud_model),
            embed_model=os.environ.get("EXPERTS_EMBED_MODEL", cls.embed_model),
            fim_model=os.environ.get("EXPERTS_FIM_MODEL", cls.fim_model),
            ui_port=int(os.environ.get("EXPERTS_UI_PORT", cls.ui_port)),
            data_dir=Path(
                os.environ.get("EXPERTS_DATA_DIR", Path.home() / ".local" / "share" / "experts")
            ).expanduser(),
        )
