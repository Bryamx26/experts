"""Exécution de commande, toujours soumise à ta confirmation.

Sécurité :
- confirmation obligatoire à chaque appel (la fonction `confirm` est injectée,
  donc impossible à contourner par le modèle) ;
- pas de shell : shlex.split + subprocess sans `shell=True`, donc pas de
  `;`, `|`, `$(...)` interprétés ;
- délai maximal et dossier de travail fixé à l'espace de travail.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from experts.tools.base import Tool, ToolError

ConfirmFn = Callable[[list[str]], bool]


def make_run_command(workspace: Path, confirm: ConfirmFn, *, timeout: float = 60.0) -> Tool:
    def run_command(command: str) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"Commande mal formée : {exc}") from exc
        if not argv:
            raise ToolError("Commande vide.")
        if not confirm(argv):
            return "Refusé par l'utilisateur. Ne relance pas cette commande ; propose une autre piste."
        try:
            proc = subprocess.run(
                argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"Programme introuvable : {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Délai de {timeout:.0f} s dépassé.") from exc
        return (
            f"code de sortie : {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    return Tool(
        name="run_command",
        description=(
            "Exécute une commande dans l'espace de travail, après confirmation de l'utilisateur. "
            "Pas de shell : pas de tubes, redirections ni variables."
        ),
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Ex. : 'ruff check app.py'."}},
            "required": ["command"],
        },
        func=run_command,
    )
