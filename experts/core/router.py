"""Routeur : choisit le spécialiste selon le fichier édité.

Ordre, du moins cher au plus cher :
1. règles des YAML (nom de fichier, puis clés YAML de premier niveau comme
   apiVersion + kind) ; la règle la plus précise gagne ;
2. mots-clés du contenu, si un compagnon se détache nettement ;
3. classification par le modèle, seulement si c'est encore ambigu ;
4. compagnon par défaut.
Le modèle n'est appelé qu'en dernier recours : c'est lent et ce n'est pas fiable à 100 %.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import yaml

from experts.core.agent import ChatModel
from experts.core.companions import CompanionSpec
from experts.core.llm import LLMError, Message

Reason = Literal["règle", "mots-clés", "modèle", "défaut"]
Classifier = Callable[[str, str, list[str]], str | None]
CLASSIFY_CHARS = 2_000


@dataclass(frozen=True, slots=True)
class Route:
    companion: CompanionSpec
    reason: Reason


def top_level_keys(content: str) -> set[str]:
    keys: set[str] = set()
    try:
        for doc in yaml.safe_load_all(content):
            if isinstance(doc, dict):
                keys |= {str(k) for k in doc}
    except yaml.YAMLError:
        # YAML cassé en cours de frappe : on se rabat sur une lecture ligne à ligne.
        keys |= set(re.findall(r"^([A-Za-z_][\w-]*):", content, re.MULTILINE))
    return keys


class Router:
    def __init__(
        self,
        companions: Mapping[str, CompanionSpec],
        *,
        default: str = "python",
        classifier: Classifier | None = None,
    ) -> None:
        if default not in companions:
            raise ValueError(f"Compagnon par défaut inconnu : {default}")
        self.companions = dict(companions)
        self.default = default
        self._classifier = classifier

    def by_rules(self, filename: str, content: str) -> CompanionSpec | None:
        keys: set[str] | None = None
        best: tuple[int, CompanionSpec | None] = (0, None)
        tie = False
        for spec in self.companions.values():
            for rule in spec.match:
                score = rule.specificity(filename)
                if score and rule.contains_keys:
                    keys = top_level_keys(content) if keys is None else keys
                    if not set(rule.contains_keys) <= keys:
                        score = 0
                if score > best[0]:
                    best, tie = (score, spec), False
                elif score and score == best[0] and spec is not best[1]:
                    tie = True
        return None if tie else best[1]

    def by_keywords(self, content: str) -> CompanionSpec | None:
        scores = sorted(
            ((sum(content.count(k) for k in s.keywords), s) for s in self.companions.values()),
            key=lambda t: t[0],
            reverse=True,
        )
        if not scores or scores[0][0] < 2:
            return None
        if len(scores) > 1 and scores[0][0] < 2 * scores[1][0]:
            return None  # pas assez net
        return scores[0][1]

    def route(self, filename: str, content: str) -> Route:
        if spec := self.by_rules(filename, content):
            return Route(spec, "règle")
        if spec := self.by_keywords(content):
            return Route(spec, "mots-clés")
        if self._classifier is not None:
            name = self._classifier(filename, content[:CLASSIFY_CHARS], list(self.companions))
            if name in self.companions:
                return Route(self.companions[name], "modèle")
        return Route(self.companions[self.default], "défaut")


def llm_classifier(llm: ChatModel) -> Classifier:
    """Classifieur de dernier recours : une seule réponse, contrainte à la liste."""

    def classify(filename: str, excerpt: str, names: list[str]) -> str | None:
        prompt = (
            f"Fichier : {filename}\n---\n{excerpt}\n---\n"
            f"Quel spécialiste convient ? Réponds par un seul mot parmi : {', '.join(names)}, aucun."
        )
        try:
            answer = llm.chat([Message(role="user", content=prompt)]).content.strip().lower()
        except LLMError:
            return None
        word = re.sub(r"[^a-z]", "", answer.split()[0]) if answer else ""
        return word if word in names else None

    return classify
