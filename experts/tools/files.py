"""Lecture de fichiers, confinée à l'espace de travail.

Sécurité : le chemin est résolu (liens symboliques compris) puis comparé à la
racine, ce qui bloque `../../etc/shadow` et les liens qui sortent du dossier.
"""

from __future__ import annotations

from pathlib import Path

from experts.tools.base import Tool, ToolError

MAX_FILE_BYTES = 200_000
# Jamais envoyés au modèle, qu'il soit local ou dans le cloud.
SECRET_NAMES = {".env", ".netrc", ".pgpass", "id_rsa", "id_ed25519", "id_ecdsa", "credentials"}
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".kdbx")


def is_secret(path: Path) -> bool:
    name = path.name.lower()
    return name in SECRET_NAMES or name.startswith(".env.") or name.endswith(SECRET_SUFFIXES)


def resolve_inside(workspace: Path, relative: str) -> Path:
    root = workspace.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"'{relative}' est hors de l'espace de travail.")
    return target


def make_read_file(workspace: Path) -> Tool:
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        target = resolve_inside(workspace, path)
        if not target.is_file():
            raise ToolError(f"'{path}' n'existe pas ou n'est pas un fichier.")
        if is_secret(target):
            raise ToolError(f"'{path}' ressemble à un fichier de secrets ; lecture refusée.")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"'{path}' dépasse {MAX_FILE_BYTES} octets ; précise une plage de lignes.")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(start_line, 1)
        end = min(end_line or len(lines), len(lines))
        # Numéros de ligne pour que le modèle cite la ligne exacte.
        return "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(start, end + 1))

    return Tool(
        name="read_file",
        description="Lit un fichier texte de l'espace de travail, avec numéros de ligne.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Chemin relatif à l'espace de travail."},
                "start_line": {"type": "integer", "description": "Première ligne (1 par défaut)."},
                "end_line": {"type": "integer", "description": "Dernière ligne incluse."},
            },
            "required": ["path"],
        },
        func=read_file,
    )


def make_list_files(workspace: Path) -> Tool:
    def list_files(path: str = ".") -> str:
        target = resolve_inside(workspace, path)
        if not target.is_dir():
            raise ToolError(f"'{path}' n'est pas un dossier.")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        visible = [e for e in entries if not e.name.startswith(".")][:200]
        return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in visible) or "(vide)"

    return Tool(
        name="list_files",
        description="Liste le contenu d'un dossier de l'espace de travail.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Dossier relatif ('.' par défaut)."}},
        },
        func=list_files,
    )
