"""Assemblage : réglages, modèle, compagnons, routeur et outils.

Partagé par le terminal (étape 1), le serveur LSP (étape 4) et l'interface
(étapes 5 et 7), pour que tous voient le même compagnon pour le même fichier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from experts.core.agent import Agent, ChatModel
from experts.core.cloud import AnthropicClient
from experts.core.companions import CompanionSpec, load_companions
from experts.core.config import Settings
from experts.core.llm import OllamaClient, ToolCall
from experts.core.rag import DocStore, OllamaEmbedder, make_search_docs
from experts.core.router import Route, Router, llm_classifier
from experts.tools.base import ToolRegistry
from experts.tools.files import make_list_files, make_read_file
from experts.tools.shell import ConfirmFn, make_run_command
from experts.tools.validators import Validator, from_names, make_validate_file
from experts.tools.web_search import make_web_search

MAX_ROUTE_BYTES = 200_000


def make_llm(settings: Settings) -> ChatModel:
    if settings.backend == "anthropic":
        return AnthropicClient(settings.cloud_model)
    return OllamaClient(settings.ollama_host, settings.chat_model, timeout=settings.request_timeout)


@dataclass
class Runtime:
    settings: Settings
    llm: ChatModel
    companions: dict[str, CompanionSpec]
    router: Router
    tools: ToolRegistry

    def route_file(self, path: Path, content: str | None = None) -> Route:
        if content is None:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[:MAX_ROUTE_BYTES]
            except OSError:
                content = ""
        return self.router.route(path.name, content)

    def validators_for(self, path: Path, content: str | None = None) -> list[Validator]:
        spec = self.route_file(path, content).companion
        return from_names(spec.validator_names_for(path.name))

    def agent_for(self, spec: CompanionSpec, on_tool_call: Callable[[ToolCall], None] | None = None) -> Agent:
        return Agent(
            self.llm,
            self.tools.subset(spec.tools),
            system_prompt=spec.full_prompt,
            max_tool_rounds=self.settings.max_tool_rounds,
            on_tool_call=on_tool_call,
        )

    def close(self) -> None:
        self.llm.close()


def build_runtime(settings: Settings, confirm: ConfirmFn, *, llm: ChatModel | None = None) -> Runtime:
    llm = llm or make_llm(settings)
    companions = load_companions()
    router = Router(companions, classifier=llm_classifier(llm))
    runtime = Runtime(settings, llm, companions, router, ToolRegistry())
    store = DocStore(settings.data_dir / "chroma", OllamaEmbedder(settings.ollama_host, settings.embed_model))
    for tool in (
        make_read_file(settings.workspace),
        make_list_files(settings.workspace),
        make_validate_file(settings.workspace, runtime.validators_for),
        make_search_docs(store),
        make_web_search(settings.searxng_url),
        make_run_command(settings.workspace, confirm),
    ):
        runtime.tools.register(tool)
    return runtime
