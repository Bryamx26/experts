"""Serveur LSP des experts (pygls).

- Diagnostics : validateurs déterministes après ~500 ms sans frappe, ou tout de
  suite à l'ouverture et à la sauvegarde. Croix rouge sur la ligne exacte.
- Code actions : « Corriger avec l'expert » sur chaque diagnostic. La correction
  n'est calculée qu'au moment où tu choisis l'action (codeAction/resolve), car
  un appel au modèle prend des secondes ; elle est revalidée avant d'être
  proposée.

Les validateurs et le modèle sont bloquants : ils tournent dans un thread
(`asyncio.to_thread`) pour que l'éditeur reste fluide.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer
from pygls.uris import to_fs_path

from experts.core.app import Runtime, build_runtime
from experts.core.bridge import BridgeClient
from experts.core.companions import CompanionSpec
from experts.core.config import Settings
from experts.core.fixer import minimal_edit, propose_fix
from experts.tools.validators import Diagnostic, Validator, from_names, validate_content, validate_path

DEBOUNCE_S = 0.5
log = logging.getLogger("experts.lsp")
SEVERITY = {
    "error": lsp.DiagnosticSeverity.Error,
    "warning": lsp.DiagnosticSeverity.Warning,
    "info": lsp.DiagnosticSeverity.Information,
}
Event = dict[str, Any]


def to_lsp(d: Diagnostic, lines: list[str], companion: str) -> lsp.Diagnostic:
    line = max(d.line - 1, 0)
    end_col = len(lines[line]) if line < len(lines) else 0
    return lsp.Diagnostic(
        range=lsp.Range(lsp.Position(line, max(d.column - 1, 0)), lsp.Position(line, max(end_col, d.column))),
        message=d.message,
        severity=SEVERITY[d.severity],
        code=d.code,
        source=f"experts/{d.source}",
        data={"companion": companion, "line": d.line, "column": d.column, "severity": d.severity},
    )


def from_lsp(d: lsp.Diagnostic, filename: str) -> Diagnostic:
    data = d.data if isinstance(d.data, dict) else {}
    return Diagnostic(
        file=filename,
        line=int(data.get("line", d.range.start.line + 1)),
        column=int(data.get("column", d.range.start.character + 1)),
        severity=data.get("severity", "error"),
        code=str(d.code),
        message=d.message,
        source=(d.source or "").removeprefix("experts/"),
    )


class ExpertsServer(LanguageServer):
    def __init__(self) -> None:
        super().__init__("experts", "0.4.0")
        self.runtime: Runtime | None = None
        self.bridge: BridgeClient | None = None
        self._pending: dict[str, asyncio.Task[None]] = {}
        # Branché sur la fenêtre du compagnon à l'étape 5 (WebSocket).
        self.emit: Callable[[Event], None] = lambda event: None

    def setup(self, root: Path | None) -> None:
        settings = Settings.from_env()
        if root is not None:
            settings = replace(settings, workspace=root.resolve())
        # Pas de terminal ici : toute commande est refusée tant que la fenêtre (étape 5) ne confirme pas.
        self.runtime = build_runtime(settings, confirm=lambda argv: False)

    def context(self, uri: str) -> tuple[Path, str, CompanionSpec, list[Validator]]:
        assert self.runtime is not None
        doc = self.workspace.get_text_document(uri)
        path = Path(to_fs_path(uri) or uri)
        spec = self.runtime.route_file(path, doc.source).companion
        return path, doc.source, spec, from_names(spec.validator_names_for(path.name))

    async def validate(self, uri: str) -> None:
        path, source, spec, validators = self.context(uri)
        self.emit({"type": "focus", "companion": spec.name, "file": str(path)})
        if not validators:
            self.text_document_publish_diagnostics(lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[]))
            return
        self.emit({"type": "state", "state": "think"})
        on_disk = path.is_file() and path.read_text(encoding="utf-8", errors="replace") == source
        # Fichier identique au disque : on valide le vrai fichier pour profiter de la config du projet
        # (pyproject.toml, .hadolint.yaml…). Sinon, copie temporaire du texte en cours.
        found = await asyncio.to_thread(
            (lambda: validate_path(path, validators))
            if on_disk
            else (lambda: validate_content(path.name, source, validators))
        )
        lines = source.splitlines()
        self.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[to_lsp(d, lines, spec.name) for d in found])
        )
        errors = [d for d in found if d.severity == "error"]
        self.emit(
            {
                "type": "diagnostics",
                "companion": spec.name,
                "file": str(path),
                "items": [
                    {"line": d.line, "severity": d.severity, "code": d.code, "message": d.message}
                    for d in found
                ],
            }
        )
        self.emit({"type": "state", "state": "alert" if errors else "idle"})

    def schedule(self, uri: str, delay: float) -> None:
        if task := self._pending.pop(uri, None):
            task.cancel()

        async def run() -> None:
            await asyncio.sleep(delay)
            try:
                await self.validate(uri)
            except Exception:
                log.exception("Validation impossible pour %s", uri)

        self._pending[uri] = asyncio.ensure_future(run())


server = ExpertsServer()


@server.feature(lsp.INITIALIZE)
def initialize(ls: ExpertsServer, params: lsp.InitializeParams) -> None:
    root: Path | None = None
    if params.workspace_folders:
        root = Path(to_fs_path(params.workspace_folders[0].uri) or ".")
    elif params.root_uri:
        root = Path(to_fs_path(params.root_uri) or ".")
    ls.setup(root)


@server.feature(lsp.INITIALIZED)
async def initialized(ls: ExpertsServer, params: lsp.InitializedParams) -> None:
    # Liaison avec la fenêtre du compagnon (étape 5) ; sans fenêtre, l'éditeur fonctionne pareil.
    assert ls.runtime is not None
    ls.bridge = BridgeClient(ls.runtime.settings.data_dir)
    ls.bridge.start()
    ls.emit = ls.bridge.emit


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: ExpertsServer, params: lsp.DidOpenTextDocumentParams) -> None:
    ls.schedule(params.text_document.uri, 0)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: ExpertsServer, params: lsp.DidChangeTextDocumentParams) -> None:
    ls.schedule(params.text_document.uri, DEBOUNCE_S)


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: ExpertsServer, params: lsp.DidSaveTextDocumentParams) -> None:
    ls.schedule(params.text_document.uri, 0)


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: ExpertsServer, params: lsp.DidCloseTextDocumentParams) -> None:
    if task := ls._pending.pop(params.text_document.uri, None):
        task.cancel()
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
    )


@server.feature(
    lsp.TEXT_DOCUMENT_CODE_ACTION,
    lsp.CodeActionOptions(code_action_kinds=[lsp.CodeActionKind.QuickFix], resolve_provider=True),
)
def code_action(ls: ExpertsServer, params: lsp.CodeActionParams) -> list[lsp.CodeAction]:
    assert ls.runtime is not None
    actions: list[lsp.CodeAction] = []
    for diag in params.context.diagnostics:
        if (
            not (diag.source or "").startswith("experts/")
            or diag.severity == lsp.DiagnosticSeverity.Information
        ):
            continue
        name = diag.data.get("companion") if isinstance(diag.data, dict) else None
        spec = ls.runtime.companions.get(str(name)) or ls.runtime.companions[ls.runtime.router.default]
        actions.append(
            lsp.CodeAction(
                title=f"{spec.display_name} : corriger « {diag.code} »",
                kind=lsp.CodeActionKind.QuickFix,
                diagnostics=[diag],
                data={"uri": params.text_document.uri},
            )
        )
    return actions


@server.feature(lsp.CODE_ACTION_RESOLVE)
async def code_action_resolve(ls: ExpertsServer, action: lsp.CodeAction) -> lsp.CodeAction:
    assert ls.runtime is not None and action.diagnostics
    uri = action.data["uri"] if isinstance(action.data, dict) else ""
    path, source, spec, validators = ls.context(uri)
    target = from_lsp(action.diagnostics[0], path.name)
    ls.emit({"type": "state", "state": "think"})
    before = await asyncio.to_thread(validate_content, path.name, source, validators)
    fix = await asyncio.to_thread(
        propose_fix, ls.runtime.llm, spec, path.name, source, target, validators, before
    )
    if fix is None:
        ls.emit({"type": "state", "state": "alert"})
        ls.window_show_message(
            lsp.ShowMessageParams(
                type=lsp.MessageType.Warning,
                message=f"{spec.display_name} n'a pas trouvé de correction validée pour {target.code}.",
            )
        )
        return action
    edit = minimal_edit(source, fix.content)
    if edit is not None:
        rng = lsp.Range(lsp.Position(edit.start_line, 0), lsp.Position(edit.end_line, 0))
        action.edit = lsp.WorkspaceEdit(changes={uri: [lsp.TextEdit(range=rng, new_text=edit.text)]})
    ls.emit({"type": "suggestion", "companion": spec.name, "text": fix.explanation})
    ls.emit({"type": "state", "state": "suggest"})
    return action
