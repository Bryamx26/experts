"""Étape 7 : questions posées en cliquant sur l'expert.

Flux (la fenêtre parle au serveur LSP actif par le WebSocket local) :
- `ask` → l'expert du fichier répond, avec le contexte joint automatiquement
  (fichier, fonction, lignes autour du curseur) et la liste des sources utilisées ;
- `confirm` / `confirm_reply` : si le modèle veut lancer une commande, la fenêtre
  te la montre et attend ton accord (refus par défaut au bout de 2 minutes) ;
- `insert_preview` → `diff` → `insert_apply` : « Insérer » montre d'abord le diff,
  et rien n'est écrit dans l'éditeur sans ta confirmation.

L'agent est bloquant : il tourne dans un thread, et tout ce qu'il envoie repasse
par la boucle asyncio (`call_soon_threadsafe`).
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experts.core.agent import Agent
from experts.core.app import Runtime
from experts.core.fixer import TextEdit
from experts.core.llm import LLMError, ToolCall

CONTEXT_LINES = 15
CONFIRM_TIMEOUT_S = 120.0
Event = dict[str, Any]


@dataclass(frozen=True, slots=True)
class EditorContext:
    uri: str
    path: Path
    text: str
    start: int  # lignes 0-based, incluses
    end: int
    symbol: str | None = None

    @property
    def label(self) -> str:
        where = f"{self.path.name} › {self.symbol}" if self.symbol else self.path.name  # noqa: RUF001
        return f"{where} · l. {self.start + 1}-{self.end + 1}"

    def snippet(self) -> str:
        lines = self.text.splitlines()
        return "\n".join(f"{n + 1:>4} | {lines[n]}" for n in range(self.start, min(self.end + 1, len(lines))))


def enclosing_python_symbol(text: str, line: int) -> tuple[str, int, int] | None:
    """(nom qualifié, début, fin) 0-based de la fonction/classe la plus interne autour de `line`."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    best: tuple[str, int, int] | None = None

    def visit(node: ast.AST, prefix: str) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                start, end = child.lineno - 1, (child.end_lineno or child.lineno) - 1
                name = f"{prefix}{child.name}"
                if start <= line <= end:
                    best = (name, start, end)
                    visit(child, name + ".")

    visit(tree, "")
    return best


def build_context(uri: str, path: Path, text: str, cursor: tuple[int, int] | None) -> EditorContext:
    total = max(len(text.splitlines()), 1)
    if cursor is None:
        return EditorContext(uri, path, text, 0, min(total, 2 * CONTEXT_LINES) - 1)
    first, last = cursor
    if path.suffix == ".py" and (sym := enclosing_python_symbol(text, first)):
        name, start, end = sym
        if end - start <= 4 * CONTEXT_LINES:
            return EditorContext(uri, path, text, start, end, name)
    return EditorContext(uri, path, text, max(first - CONTEXT_LINES, 0), min(last + CONTEXT_LINES, total - 1))


def source_label(call: ToolCall) -> str | None:
    args = call.arguments
    if call.name == "search_docs":
        return f"docs {args.get('collection', '')} (RAG)"
    if call.name == "web_search":
        return f"SearXNG · {str(args.get('query', ''))[:40]}"
    if call.name == "read_file":
        return f"fichier · {args.get('path', '')}"
    if call.name == "validate_file":
        return "validateurs"
    return None


def apply_text_edit(text: str, edit: TextEdit) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[: edit.start_line]) + edit.text + "".join(lines[edit.end_line :])


def insertion_edit(ctx: EditorContext, code: str, selection: tuple[int, int] | None) -> TextEdit:
    """Remplace la sélection (lignes entières) par le code, ou l'insère sous la ligne du curseur."""
    if not code.endswith("\n"):
        code += "\n"
    if selection is not None and selection[1] > selection[0]:
        return TextEdit(selection[0], selection[1] + 1, code)
    line = (selection[0] + 1) if selection else len(ctx.text.splitlines())
    return TextEdit(line, line, code)


def unified_diff(path: Path, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{path.name}", f"b/{path.name}"
        )
    )


@dataclass
class AskService:
    runtime: Runtime
    send: Callable[[Event], None]  # appelé depuis la boucle asyncio
    get_context: Callable[[], EditorContext | None]
    get_selection: Callable[[str], tuple[int, int] | None]
    apply_edit: Callable[[str, TextEdit], Awaitable[bool]]
    loop: asyncio.AbstractEventLoop | None = None
    agents: dict[str, Agent] = field(default_factory=dict)
    _confirms: dict[str, tuple[threading.Event, list[bool]]] = field(default_factory=dict)
    _previews: dict[str, tuple[str, TextEdit, str]] = field(default_factory=dict)

    def _send_threadsafe(self, event: Event) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.send, event)

    def confirm(self, argv: list[str]) -> bool:
        """Appelé par run_command, dans le thread de l'agent : attend la réponse de la fenêtre."""
        cid = uuid.uuid4().hex
        done, answer = threading.Event(), [False]
        self._confirms[cid] = (done, answer)
        self._send_threadsafe({"type": "confirm", "id": cid, "command": " ".join(argv)})
        done.wait(CONFIRM_TIMEOUT_S)
        self._confirms.pop(cid, None)
        return answer[0]

    async def handle(self, msg: Event) -> None:
        kind = msg.get("type")
        if kind == "ask":
            await self._ask(str(msg.get("id", "")), str(msg.get("question", "")).strip())
        elif kind == "confirm_reply":
            entry = self._confirms.get(str(msg.get("id")))
            if entry is not None:
                entry[1][0] = msg.get("ok") is True  # tout sauf un vrai « oui » vaut refus
                entry[0].set()
        elif kind == "insert_preview":
            self._preview(str(msg.get("id", "")), str(msg.get("code", "")))
        elif kind == "insert_apply":
            await self._apply(str(msg.get("id", "")))

    async def _ask(self, qid: str, question: str) -> None:
        self.loop = asyncio.get_running_loop()
        ctx = self.get_context()
        if not question:
            return
        if ctx is None:
            spec = self.runtime.companions[self.runtime.router.default]
            prompt, label = question, "aucun fichier ouvert"
        else:
            spec = self.runtime.route_file(ctx.path, ctx.text).companion
            prompt = (
                f"Contexte joint automatiquement : {ctx.path.name}, lignes {ctx.start + 1} à {ctx.end + 1}"
                f"{f' (dans {ctx.symbol})' if ctx.symbol else ''}.\n```\n{ctx.snippet()}\n```\n\n"
                f"Question : {question}"
            )
            label = ctx.label
        sources: list[str] = []

        def on_tool(call: ToolCall) -> None:
            if (s := source_label(call)) and s not in sources:
                sources.append(s)

        if spec.name not in self.agents:
            self.agents[spec.name] = self.runtime.agent_for(spec)
        agent = self.agents[spec.name]
        agent._on_tool_call = on_tool
        self.send({"type": "state", "state": "think"})
        try:
            text = await asyncio.to_thread(agent.ask, prompt)
        except LLMError as exc:
            text = f"Je n'arrive pas à joindre le modèle : {exc}"
        self.send(
            {
                "type": "answer",
                "id": qid,
                "companion": spec.name,
                "context": label,
                "text": text,
                "sources": sources,
            }
        )
        self.send({"type": "state", "state": "idle"})

    def _preview(self, pid: str, code: str) -> None:
        ctx = self.get_context()
        if ctx is None or not code.strip():
            self.send(
                {"type": "diff", "id": pid, "diff": "", "error": "Aucun fichier ouvert dans l'éditeur."}
            )
            return
        edit = insertion_edit(ctx, code, self.get_selection(ctx.uri))
        new_text = apply_text_edit(ctx.text, edit)
        self._previews[pid] = (ctx.uri, edit, ctx.text)
        self.send(
            {
                "type": "diff",
                "id": pid,
                "file": ctx.path.name,
                "diff": unified_diff(ctx.path, ctx.text, new_text),
            }
        )

    async def _apply(self, pid: str) -> None:
        preview = self._previews.pop(pid, None)
        ctx = self.get_context()
        if preview is None or ctx is None or ctx.uri != preview[0] or ctx.text != preview[2]:
            # Le fichier a changé depuis l'aperçu : on n'écrit pas un diff périmé.
            self.send(
                {"type": "suggestion", "text": "Le fichier a changé depuis l'aperçu ; redemande l'insertion."}
            )
            return
        ok = await self.apply_edit(preview[0], preview[1])
        self.send(
            {"type": "suggestion", "text": "Code inséré." if ok else "L'éditeur a refusé la modification."}
        )
