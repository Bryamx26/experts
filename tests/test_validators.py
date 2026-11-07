"""Validateurs : tests réels quand l'outil est installé, sinon ignorés."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from experts.tools import validators as v
from experts.tools.base import ToolRegistry


def needs(binary: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(shutil.which(binary) is None, reason=f"{binary} absent")


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_routing(tmp_path: Path) -> None:
    k8s = write(tmp_path, "deploy.yaml", "apiVersion: v1\nkind: Pod\n")
    plain = write(tmp_path, "config.yaml", "a: 1\n")
    assert v.validators_for(tmp_path / "app.py") == [v.RUFF]
    assert v.validators_for(tmp_path / "Dockerfile") == [v.HADOLINT]
    assert v.validators_for(tmp_path / "api.dockerfile") == [v.HADOLINT]
    assert v.validators_for(tmp_path / "docker-compose.prod.yml") == [v.YAMLLINT, v.COMPOSE]
    assert v.validators_for(k8s) == [v.YAMLLINT, v.KUBECONFORM]
    assert v.validators_for(plain) == [v.YAMLLINT]
    assert v.validators_for(tmp_path / "notes.md") == []


@needs("ruff")
def test_ruff(tmp_path: Path) -> None:
    diags = v.validate_path(write(tmp_path, "app.py", "import os\n\nprint('ok')\n"))
    assert [(d.line, d.code, d.severity) for d in diags] == [(1, "F401", "error")]


@needs("ruff")
def test_ruff_syntax_error(tmp_path: Path) -> None:
    diags = v.validate_path(write(tmp_path, "app.py", "def f(:\n    pass\n"))
    assert diags and diags[0].line == 1 and diags[0].severity == "error"


@needs("hadolint")
def test_hadolint(tmp_path: Path) -> None:
    diags = v.validate_path(write(tmp_path, "Dockerfile", "FROM ubuntu\nRUN apt-get install curl\n"))
    assert ("DL3006", 1) in [(d.code, d.line) for d in diags]
    assert all(d.source == "hadolint" for d in diags)


@needs("yamllint")
def test_yamllint_duplicate_key(tmp_path: Path) -> None:
    diags = v.validate_path(write(tmp_path, "config.yaml", "a: 1\nb: 2\na: 3\n"))
    assert [(d.line, d.code) for d in diags] == [(3, "key-duplicates")]


def test_kubeconform_points_to_exact_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Analyse d'une sortie kubeconform réelle (rejouée) : le chemin JSON devient une ligne."""
    manifest = (
        "apiVersion: v1\nkind: Service\nmetadata:\n  name: autre\n---\n"
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\nspec:\n  replicas: '3'\n"
    )
    path = write(tmp_path, "k.yaml", manifest)
    output = {
        "resources": [
            {
                "filename": "k.yaml",
                "kind": "Deployment",
                "name": "web",
                "status": "statusInvalid",
                "validationErrors": [
                    {"path": "/spec/replicas", "msg": "expected integer or null, but got string"}
                ],
            }
        ]
    }

    def fake_exec(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(v, "_exec", fake_exec)
    diags = v.run_kubeconform(path)
    assert [(d.line, d.source) for d in diags] == [(11, "kubeconform")]


@needs("docker")
def test_compose_unknown_key_line(tmp_path: Path) -> None:
    if subprocess.run(["docker", "compose", "version"], capture_output=True, check=False).returncode:
        pytest.skip("docker compose absent")
    path = write(tmp_path, "compose.yaml", "services:\n  web:\n    image: nginx\n    imagee: nginx\n")
    diags = v.run_compose(path)
    assert [(d.line, d.severity) for d in diags] == [(4, "error")]
    assert "imagee" in diags[0].message


@needs("docker")
def test_compose_parse_error_line(tmp_path: Path) -> None:
    if subprocess.run(["docker", "compose", "version"], capture_output=True, check=False).returncode:
        pytest.skip("docker compose absent")
    path = write(tmp_path, "compose.yaml", "services:\n  web:\n    image: nginx\n   ports: [80]\n")
    diags = v.run_compose(path)
    assert len(diags) == 1 and diags[0].line in {2, 4}


@needs("hadolint")
def test_validate_content_for_llm_fix() -> None:
    """Une correction proposée par le LLM repasse les validateurs avant affichage."""
    bad = v.validate_content("Dockerfile", "FROM ubuntu\n")
    good = v.validate_content("Dockerfile", "FROM ubuntu:24.04\n")
    assert any(d.code == "DL3006" for d in bad)
    assert not any(d.code == "DL3006" for d in good)


def test_missing_binary_is_reported_not_raised(tmp_path: Path) -> None:
    ghost = v.Validator("fantome", "binaire-qui-nexiste-pas", lambda p: [])
    diags = v.validate_path(write(tmp_path, "app.py", "x = 1\n"), [ghost])
    assert [(d.severity, d.code) for d in diags] == [("info", "absent")]


@needs("ruff")
def test_validate_file_tool(tmp_path: Path) -> None:
    write(tmp_path, "app.py", "import os\n")
    registry = ToolRegistry([v.make_validate_file(tmp_path)])
    assert "app.py:1:8 [error] F401" in registry.call("validate_file", {"path": "app.py"})
    assert "hors de l'espace" in registry.call("validate_file", {"path": "../x.py"})
