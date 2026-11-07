"""Compagnon en terminal. Lancement : python -m experts [--file Dockerfile]"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experts.core.agent import Agent
from experts.core.app import Runtime, build_runtime, make_llm
from experts.core.companions import CompanionSpec
from experts.core.config import Settings
from experts.core.llm import LLMError, ToolCall

HELP = (
    "Commandes : /fichier <chemin> (choisit l'expert d'après le fichier), /expert <nom>, "
    "/reset, /outils, /quitter"
)
CLOUD_WARNING = """\
  ATTENTION : mode cloud. Tes questions et le contenu des fichiers lus par le
  compagnon partent chez Anthropic (payant, hors de ta machine).
  Les fichiers de secrets (.env, clés…) restent bloqués."""


def confirm_in_terminal(argv: list[str]) -> bool:
    print(f"\n  Le compagnon veut exécuter : {' '.join(argv)}")
    return input("  Confirmer ? [o/N] ").strip().lower() in {"o", "oui"}


def show_tool_call(call: ToolCall) -> None:
    print(f"  · {call.name}({call.arguments})")


def parse_settings() -> tuple[Settings, Path | None]:
    parser = argparse.ArgumentParser(description="Compagnon de code en terminal.")
    parser.add_argument("--model", help="Modèle Ollama (défaut : qwen3.5:4b).")
    parser.add_argument("--workspace", type=Path, help="Dossier accessible au compagnon.")
    parser.add_argument("--file", type=Path, help="Fichier édité : choisit le bon expert.")
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Utilise un modèle payant (Claude, clé ANTHROPIC_API_KEY) au lieu d'Ollama.",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.model:
        settings = replace(settings, chat_model=args.model)
    if args.workspace:
        settings = replace(settings, workspace=args.workspace.resolve())
    if args.cloud:
        settings = replace(settings, backend="anthropic")
    return settings, args.file


class Session:
    """Un agent par expert, gardé en mémoire : changer d'expert ne perd pas les conversations."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._agents: dict[str, Agent] = {}
        self.current = runtime.companions[runtime.router.default]

    @property
    def agent(self) -> Agent:
        if self.current.name not in self._agents:
            self._agents[self.current.name] = self.runtime.agent_for(self.current, show_tool_call)
        return self._agents[self.current.name]

    def switch(self, spec: CompanionSpec, why: str) -> None:
        self.current = spec
        print(f"Expert actif : {spec.display_name} ({spec.character}) — {why}")

    def switch_for_file(self, path: Path) -> None:
        full = path if path.is_absolute() else self.runtime.settings.workspace / path
        route = self.runtime.route_file(full)
        self.switch(route.companion, f"choisi par {route.reason} pour {path.name}")


def main() -> None:
    settings, file = parse_settings()
    if settings.backend == "anthropic":
        print(CLOUD_WARNING)
        if input("  Continuer en mode cloud ? [o/N] ").strip().lower() not in {"o", "oui"}:
            return
    try:
        runtime = build_runtime(settings, confirm_in_terminal, llm=make_llm(settings))
    except LLMError as exc:
        print(f"[erreur] {exc}")
        return
    session = Session(runtime)
    model = settings.cloud_model if settings.backend == "anthropic" else settings.chat_model
    print(f"Experts ({model}) — espace de travail : {settings.workspace}")
    print(HELP)
    if file:
        session.switch_for_file(file)
    else:
        session.switch(session.current, "par défaut")
    try:
        while True:
            try:
                question = input(f"\nToi ({session.current.name}) > ").strip()
            except EOFError:
                break
            command, _, arg = question.partition(" ")
            if not question:
                continue
            if command in {"/quitter", "/q", "/exit"}:
                break
            if command == "/reset":
                session.agent.reset()
                print("Conversation effacée.")
            elif command == "/outils":
                print(", ".join(session.agent.tool_names))
            elif command == "/fichier" and arg:
                session.switch_for_file(Path(arg.strip()))
            elif command == "/expert":
                spec = runtime.companions.get(arg.strip())
                if spec:
                    session.switch(spec, "choisi à la main")
                else:
                    print(f"Experts disponibles : {', '.join(runtime.companions)}")
            else:
                try:
                    print(f"\n{session.current.display_name} > {session.agent.ask(question)}")
                except LLMError as exc:
                    print(f"\n[erreur] {exc}")
    except KeyboardInterrupt:
        print()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
