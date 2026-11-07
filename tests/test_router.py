"""Étape 3 : YAML des compagnons et routeur."""

from __future__ import annotations

from pathlib import Path

import pytest

from experts.core.companions import CompanionError, load_companions, parse_companion
from experts.core.llm import Message
from experts.core.router import Router, llm_classifier, top_level_keys

SPECS = load_companions()


def route(filename: str, content: str = "") -> tuple[str, str]:
    r = Router(SPECS).route(filename, content)
    return r.companion.name, r.reason


def test_bundled_companions_load() -> None:
    assert set(SPECS) == {"python", "docker", "kubernetes"}
    docker = SPECS["docker"]
    assert docker.character == "la boîte"
    assert docker.sprites["idle"].name == "boite.svg"
    assert "n'invente\njamais d'erreur" in docker.full_prompt


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("app.py", "", "python"),
        ("Dockerfile", "FROM python:3.13", "docker"),
        ("Dockerfile.prod", "", "docker"),
        ("api.dockerfile", "", "docker"),
        ("compose.yaml", "services: {}", "docker"),
        ("docker-compose.override.yml", "services: {}", "docker"),
        ("deploy.yaml", "apiVersion: apps/v1\nkind: Deployment\n", "kubernetes"),
        ("multi.yml", "a: 1\n---\napiVersion: v1\nkind: Service\n", "kubernetes"),
    ],
)
def test_rules(filename: str, content: str, expected: str) -> None:
    assert route(filename, content) == (expected, "règle")


def test_broken_yaml_while_typing_still_routes() -> None:
    assert route("svc.yaml", "apiVersion: v1\nkind: Service\nspec:\n  ports: [\n") == ("kubernetes", "règle")


def test_keywords_when_no_rule() -> None:
    assert route("Makefile", "build:\n\tdocker build .\n\tdocker run --rm app\n") == ("docker", "mots-clés")


def test_plain_yaml_is_ambiguous_then_default() -> None:
    assert route("config.yaml", "debug: true\n") == ("python", "défaut")


def test_classifier_only_when_ambiguous() -> None:
    calls: list[str] = []

    def classify(filename: str, excerpt: str, names: list[str]) -> str | None:
        calls.append(filename)
        return "kubernetes"

    router = Router(SPECS, classifier=classify)
    assert router.route("Dockerfile", "").companion.name == "docker"
    assert calls == []
    r = router.route("values.yaml", "replicaCount: 2\n")
    assert (r.companion.name, r.reason) == ("kubernetes", "modèle")


def test_llm_classifier_rejects_unknown_answer() -> None:
    class Fake:
        def __init__(self, answer: str) -> None:
            self.answer = answer

        def chat(self, messages: list[Message], tools: object = None) -> Message:
            return Message(role="assistant", content=self.answer)

        def close(self) -> None:
            pass

    names = list(SPECS)
    assert llm_classifier(Fake("Kubernetes."))("values.yaml", "x", names) == "kubernetes"
    assert llm_classifier(Fake("rust"))("values.yaml", "x", names) is None


def test_validators_from_yaml() -> None:
    assert SPECS["docker"].validator_names_for("Dockerfile") == ["hadolint"]
    assert SPECS["docker"].validator_names_for("compose.yaml") == ["yamllint", "docker-compose"]
    assert SPECS["kubernetes"].validator_names_for("pod.yaml") == ["yamllint", "kubeconform"]
    assert SPECS["python"].validator_names_for("Dockerfile") == []


def test_top_level_keys() -> None:
    assert top_level_keys("a: 1\n---\nb: 2\n") == {"a", "b"}


def test_invalid_yaml_reports_file_and_field(tmp_path: Path) -> None:
    with pytest.raises(CompanionError, match=r"rust\.yaml : champ 'name'"):
        parse_companion(
            {"sprites": {s: "python/chat.svg" for s in ("idle", "think", "alert", "suggest")}},
            tmp_path / "rust.yaml",
        )
    with pytest.raises(CompanionError, match="sprites manquants pour alert"):
        parse_companion({"sprites": {"idle": "x", "think": "x", "suggest": "x"}}, tmp_path / "rust.yaml")
