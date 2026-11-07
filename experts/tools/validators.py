"""Validateurs déterministes : ruff, hadolint, yamllint, kubeconform, docker compose.

Règle du projet : les erreurs viennent de ces outils, jamais du LLM. Le LLM
explique et propose une correction, qui doit repasser `validate_content`
avant d'être affichée.

Choix :
- chaque validateur renvoie des `Diagnostic` normalisés (fichier, ligne,
  colonne, sévérité, règle), prêts pour publishDiagnostics à l'étape 4 ;
- sorties JSON quand l'outil en a une, analyse de texte sinon ;
- kubeconform et docker compose donnent un chemin (`/spec/replicas`,
  `services.web.imagee`) au lieu d'une ligne : on le retrouve dans l'arbre YAML
  pour pointer la ligne exacte ;
- outil absent = diagnostic `info`, pas de plantage : tout n'est pas installé partout ;
- ces commandes ne demandent pas de confirmation : elles sont fixées dans le
  code, en lecture seule, et le modèle ne choisit que le fichier.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from experts.tools.base import Tool, ToolError
from experts.tools.files import resolve_inside

Severity = Literal["error", "warning", "info"]
TIMEOUT_S = 30.0
# Schémas Kubernetes gardés en cache : plus rapide, et fonctionne hors ligne après le 1er passage.
KUBECONFORM_CACHE = Path.home() / ".cache" / "experts" / "kubeconform"
YAMLLINT_DEFAULT = "{extends: default, rules: {document-start: disable, line-length: {max: 120}}}"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    file: str
    line: int  # 1-based
    column: int  # 1-based
    severity: Severity
    code: str
    message: str
    source: str

    def format(self) -> str:
        where = f"{self.file}:{self.line}:{self.column}"
        return f"{where} [{self.severity}] {self.code} {self.message} ({self.source})"


@dataclass(frozen=True, slots=True)
class Validator:
    name: str
    binary: str
    run: Callable[[Path], list[Diagnostic]]


def _exec(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)


# --- Localisation d'un chemin dans un fichier YAML -----------------------------------------


def _yaml_docs(path: Path) -> list[yaml.Node]:
    try:
        return [d for d in yaml.compose_all(path.read_text(encoding="utf-8")) if d is not None]
    except yaml.YAMLError:
        return []


def _locate(node: yaml.Node, keys: Sequence[str]) -> int:
    """Ligne (1-based) de la clé la plus profonde trouvée en suivant `keys`."""
    line = node.start_mark.line + 1
    for key in keys:
        if isinstance(node, yaml.MappingNode):
            match = next((kv for kv in node.value if kv[0].value == key), None)
            if match is None:
                break
            line, node = match[0].start_mark.line + 1, match[1]
        elif isinstance(node, yaml.SequenceNode) and key.isdigit() and int(key) < len(node.value):
            node = node.value[int(key)]
            line = node.start_mark.line + 1
        else:
            break
    return line


def _scalar(node: yaml.Node, *keys: str) -> str | None:
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            return None
        found = next((v for k, v in node.value if k.value == key), None)
        if found is None:
            return None
        node = found
    return node.value if isinstance(node, yaml.ScalarNode) else None


# --- Validateurs ---------------------------------------------------------------------------


def _ruff_severity(code: str | None) -> Severity:
    # Erreurs de syntaxe et règles pyflakes (F) = vrais bugs ; le reste = style.
    if not code or code == "invalid-syntax" or code.startswith(("E9", "F")):
        return "error"
    return "warning"


def run_ruff(path: Path) -> list[Diagnostic]:
    proc = _exec(["ruff", "check", "--output-format=json", "--no-fix", path.name], path.parent)
    return [
        Diagnostic(
            file=path.name,
            line=item["location"]["row"],
            column=item["location"]["column"],
            severity=_ruff_severity(item.get("code")),
            code=item.get("code") or "invalid-syntax",
            message=item["message"],
            source="ruff",
        )
        for item in json.loads(proc.stdout or "[]")
    ]


def run_hadolint(path: Path) -> list[Diagnostic]:
    proc = _exec(["hadolint", "-f", "json", "--no-fail", path.name], path.parent)
    levels: dict[str, Severity] = {"error": "error", "warning": "warning"}
    return [
        Diagnostic(
            file=path.name,
            line=item["line"],
            column=item["column"],
            severity=levels.get(item["level"], "info"),
            code=item["code"],
            message=item["message"],
            source="hadolint",
        )
        for item in json.loads(proc.stdout or "[]")
    ]


_YAMLLINT_RE = re.compile(
    r"^(?P<f>.+?):(?P<l>\d+):(?P<c>\d+): \[(?P<s>\w+)\] (?P<m>.*?)(?: \((?P<r>[\w-]+)\))?$"
)


def run_yamllint(path: Path) -> list[Diagnostic]:
    has_config = any((path.parent / n).exists() for n in (".yamllint", ".yamllint.yaml", ".yamllint.yml"))
    argv = ["yamllint", "-f", "parsable", *([] if has_config else ["-d", YAMLLINT_DEFAULT]), path.name]
    out: list[Diagnostic] = []
    for line in _exec(argv, path.parent).stdout.splitlines():
        m = _YAMLLINT_RE.match(line)
        if m:
            sev: Severity = "error" if m["s"] == "error" else "warning"
            out.append(
                Diagnostic(path.name, int(m["l"]), int(m["c"]), sev, m["r"] or "syntax", m["m"], "yamllint")
            )
    return out


def run_kubeconform(path: Path) -> list[Diagnostic]:
    KUBECONFORM_CACHE.mkdir(parents=True, exist_ok=True)
    argv = ["kubeconform", "-output", "json", "-summary", "-strict", "-cache", str(KUBECONFORM_CACHE)]
    proc = _exec([*argv, path.name], path.parent)
    docs = _yaml_docs(path)
    out: list[Diagnostic] = []
    for res in json.loads(proc.stdout or "{}").get("resources", []):
        if res.get("status") not in {"statusInvalid", "statusError"}:
            continue
        doc = next(
            (
                d
                for d in docs
                if _scalar(d, "kind") == res.get("kind") and _scalar(d, "metadata", "name") == res.get("name")
            ),
            docs[0] if docs else None,
        )
        errors = res.get("validationErrors") or [{"path": "", "msg": res.get("msg", "")}]
        for err in errors:
            keys = [k for k in err["path"].split("/") if k]
            line = _locate(doc, keys) if doc is not None else 1
            out.append(
                Diagnostic(
                    path.name,
                    line,
                    1,
                    "error",
                    "schema",
                    f"{err['path'] or '/'} : {err['msg']}",
                    "kubeconform",
                )
            )
    return out


_COMPOSE_LINE_RE = re.compile(r"\bL(?P<l>\d+)\.C(?P<c>\d+)|line (?P<l2>\d+)")
_COMPOSE_PATH_RE = re.compile(r"validating .*?: (?P<path>[\w.\-]+) (?P<msg>.*)")


def run_compose(path: Path) -> list[Diagnostic]:
    # Sans résolution des chemins ni des variables : on valide la structure du fichier
    # lui-même, même copié dans un dossier temporaire ou sans son .env.
    argv = ["docker", "compose", "-f", path.name, "config", "-q", "--no-path-resolution", "--no-interpolate"]
    proc = _exec(argv, path.parent)
    if proc.returncode == 0:
        return []
    msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "configuration invalide"
    line, col = 1, 1
    if m := _COMPOSE_LINE_RE.search(msg):
        line, col = int(m["l"] or m["l2"]), int(m["c"] or 1)
    elif (m2 := _COMPOSE_PATH_RE.search(msg)) and (docs := _yaml_docs(path)):
        line = _locate(docs[0], m2["path"].split("."))
        msg = f"{m2['path']} : {m2['msg']}"
    return [Diagnostic(path.name, line, col, "error", "compose", msg, "docker compose")]


RUFF = Validator("ruff", "ruff", run_ruff)
HADOLINT = Validator("hadolint", "hadolint", run_hadolint)
YAMLLINT = Validator("yamllint", "yamllint", run_yamllint)
KUBECONFORM = Validator("kubeconform", "kubeconform", run_kubeconform)
COMPOSE = Validator("docker compose", "docker", run_compose)


VALIDATORS: dict[str, Validator] = {
    "ruff": RUFF,
    "hadolint": HADOLINT,
    "yamllint": YAMLLINT,
    "kubeconform": KUBECONFORM,
    "docker-compose": COMPOSE,
}
Picker = Callable[[Path], list[Validator]]

# --- Sélection et exécution ----------------------------------------------------------------


def is_compose(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".yml", ".yaml")) and (name.startswith(("docker-compose", "compose")))


def is_kubernetes(path: Path) -> bool:
    return any(_scalar(d, "apiVersion") and _scalar(d, "kind") for d in _yaml_docs(path))


def from_names(names: Sequence[str]) -> list[Validator]:
    return [VALIDATORS[n] for n in names if n in VALIDATORS]


def validators_for(path: Path) -> list[Validator]:
    """Table par défaut (sans compagnon). L'application passe par le routeur et les YAML."""
    name = path.name.lower()
    if name.endswith(".py"):
        return [RUFF]
    if name == "dockerfile" or name.startswith("dockerfile.") or name.endswith(".dockerfile"):
        return [HADOLINT]
    if is_compose(path):
        return [YAMLLINT, COMPOSE]
    if name.endswith((".yml", ".yaml")):
        return [YAMLLINT, KUBECONFORM] if is_kubernetes(path) else [YAMLLINT]
    return []


def validate_path(path: Path, validators: Sequence[Validator] | None = None) -> list[Diagnostic]:
    chosen = validators_for(path) if validators is None else validators
    out: list[Diagnostic] = []
    for v in chosen:
        if shutil.which(v.binary) is None:
            out.append(
                Diagnostic(path.name, 1, 1, "info", "absent", f"{v.binary} n'est pas installé", v.name)
            )
            continue
        try:
            out.extend(v.run(path))
        except subprocess.TimeoutExpired:
            out.append(
                Diagnostic(path.name, 1, 1, "info", "timeout", f"délai de {TIMEOUT_S:.0f} s dépassé", v.name)
            )
        except (json.JSONDecodeError, KeyError) as exc:
            out.append(Diagnostic(path.name, 1, 1, "info", "sortie", f"sortie illisible ({exc})", v.name))
    return sorted(out, key=lambda d: (d.line, d.column))


def validate_content(
    filename: str, content: str, validators: Sequence[Validator] | None = None
) -> list[Diagnostic]:
    """Valide un contenu pas encore écrit (ex. correction proposée par le LLM).

    Le fichier est copié sous le même nom dans un dossier temporaire, pour que
    les validateurs choisis par nom (Dockerfile, compose.yaml…) s'appliquent.
    """
    with tempfile.TemporaryDirectory(prefix="experts-") as tmp:
        target = Path(tmp) / Path(filename).name
        target.write_text(content, encoding="utf-8")
        return validate_path(target, validators)


def errors_only(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if d.severity == "error"]


def summarize(diagnostics: Sequence[Diagnostic]) -> str:
    return "\n".join(d.format() for d in diagnostics) or "Aucun problème détecté."


def make_validate_file(workspace: Path, pick: Picker = validators_for) -> Tool:
    def validate_file(path: str) -> str:
        target = resolve_inside(workspace, path)
        if not target.is_file():
            raise ToolError(f"'{path}' n'existe pas ou n'est pas un fichier.")
        chosen = pick(target)
        if not chosen:
            return f"Aucun validateur pour '{path}' (Python, Dockerfile, compose, YAML/Kubernetes)."
        return summarize(validate_path(target, chosen))

    return Tool(
        name="validate_file",
        description=(
            "Lance les validateurs déterministes (ruff, hadolint, yamllint, kubeconform, docker compose) "
            "sur un fichier. Source de vérité pour les erreurs : ne les invente jamais."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Chemin relatif à l'espace de travail."}
            },
            "required": ["path"],
        },
        func=validate_file,
    )
