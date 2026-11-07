"""Compagnon en terminal (étape 1). Lancement : python -m experts"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experts.core.agent import Agent
from experts.core.config import Settings
from experts.core.llm import LLMError, OllamaClient, ToolCall
from experts.tools.base import ToolRegistry
from experts.tools.files import make_list_files, make_read_file
from experts.tools.shell import make_run_command
from experts.tools.web_search import make_web_search

HELP = "Commandes : /reset (oublie la conversation), /outils, /quitter"


def confirm_in_terminal(argv: list[str]) -> bool:
    print(f"\n  Le compagnon veut exécuter : {' '.join(argv)}")
    return input("  Confirmer ? [o/N] ").strip().lower() in {"o", "oui"}


def show_tool_call(call: ToolCall) -> None:
    print(f"  · {call.name}({call.arguments})")


def build_agent(settings: Settings) -> tuple[Agent, OllamaClient]:
    llm = OllamaClient(settings.ollama_host, settings.chat_model, timeout=settings.request_timeout)
    tools = ToolRegistry(
        [
            make_read_file(settings.workspace),
            make_list_files(settings.workspace),
            make_web_search(settings.searxng_url),
            make_run_command(settings.workspace, confirm_in_terminal),
        ]
    )
    agent = Agent(llm, tools, max_tool_rounds=settings.max_tool_rounds, on_tool_call=show_tool_call)
    return agent, llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Compagnon de code en terminal.")
    parser.add_argument("--model", help="Modèle Ollama (défaut : qwen3.5:4b).")
    parser.add_argument("--workspace", type=Path, help="Dossier accessible au compagnon.")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.model:
        settings = replace(settings, chat_model=args.model)
    if args.workspace:
        settings = replace(settings, workspace=args.workspace.resolve())

    agent, llm = build_agent(settings)
    print(f"Compagnon Python ({settings.chat_model}) — espace de travail : {settings.workspace}")
    print(HELP)
    try:
        while True:
            try:
                question = input("\nToi > ").strip()
            except EOFError:
                break
            if not question:
                continue
            if question in {"/quitter", "/q", "/exit"}:
                break
            if question == "/reset":
                agent.reset()
                print("Conversation effacée.")
                continue
            if question == "/outils":
                print(", ".join(agent.tool_names))
                continue
            try:
                print(f"\nChat > {agent.ask(question)}")
            except LLMError as exc:
                print(f"\n[erreur] {exc}")
    except KeyboardInterrupt:
        print()
    finally:
        llm.close()


if __name__ == "__main__":
    main()
