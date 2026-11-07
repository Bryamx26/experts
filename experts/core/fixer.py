"""Corrections proposées par le modèle, toujours revalidées par les outils.

Principe du projet : le LLM ne détecte rien, il corrige. Une correction n'est
proposée que si les validateurs confirment que l'erreur visée a disparu et
qu'aucune nouvelle erreur n'est apparue. Sinon on redonne au modèle la sortie
des validateurs pour un second essai, puis on abandonne.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from experts.core.agent import ChatModel
from experts.core.companions import CompanionSpec
from experts.core.llm import LLMError, Message
from experts.tools.validators import Diagnostic, Validator, errors_only, summarize, validate_content

MAX_ATTEMPTS = 2
_FENCE_RE = re.compile(r"```[\w.+-]*\n(?P<code>.*?)```", re.DOTALL)

FIX_INSTRUCTIONS = """\
Corrige UNIQUEMENT le problème signalé, sans rien changer d'autre.
Réponds en deux parties :
1. une ou deux phrases en français qui expliquent l'erreur et la correction ;
2. le fichier COMPLET corrigé, dans un seul bloc de code délimité par ```.
"""


@dataclass(frozen=True, slots=True)
class Fix:
    content: str
    explanation: str
    remaining: tuple[Diagnostic, ...]


@dataclass(frozen=True, slots=True)
class TextEdit:
    """Remplacement des lignes [start, end) par `text` (lignes 0-based)."""

    start_line: int
    end_line: int
    text: str


def numbered(content: str) -> str:
    return "\n".join(f"{n:>4} | {line}" for n, line in enumerate(content.splitlines(), start=1))


def extract_code(answer: str) -> tuple[str, str] | None:
    """(explication, code) depuis la réponse du modèle, ou None."""
    blocks = list(_FENCE_RE.finditer(answer))
    if not blocks:
        return None
    last = max(blocks, key=lambda m: len(m["code"]))
    explanation = answer[: blocks[0].start()].strip() or "Correction proposée."
    return explanation, last["code"]


def _same_issue(a: Diagnostic, b: Diagnostic) -> bool:
    return a.code == b.code and a.source == b.source and abs(a.line - b.line) <= 2


def is_accepted(target: Diagnostic, before: Sequence[Diagnostic], after: Sequence[Diagnostic]) -> bool:
    if any(_same_issue(target, d) for d in after):
        return False
    return len(errors_only(after)) <= len(errors_only(before)) - (1 if target.severity == "error" else 0)


def propose_fix(
    llm: ChatModel,
    spec: CompanionSpec,
    filename: str,
    content: str,
    target: Diagnostic,
    validators: Sequence[Validator],
    before: Sequence[Diagnostic],
) -> Fix | None:
    messages = [
        Message(role="system", content=spec.system_prompt.strip() + "\n\n" + FIX_INSTRUCTIONS),
        Message(
            role="user",
            content=(
                f"Fichier {filename} :\n{numbered(content)}\n\n"
                f"Problème signalé par {target.source} : {target.format()}"
            ),
        ),
    ]
    for _ in range(MAX_ATTEMPTS):
        try:
            reply = llm.chat(messages)
        except LLMError:
            return None
        parsed = extract_code(reply.content)
        if parsed is None:
            feedback = "Je n'ai pas trouvé de bloc de code. Renvoie le fichier complet entre ```."
        else:
            explanation, code = parsed
            if content.endswith("\n") and not code.endswith("\n"):
                code += "\n"
            after = validate_content(filename, code, validators)
            if is_accepted(target, before, after):
                return Fix(code, explanation, tuple(after))
            feedback = f"Les validateurs refusent encore cette version :\n{summarize(after)}\nRecommence."
        messages += [reply, Message(role="user", content=feedback)]
    return None


def minimal_edit(old: str, new: str) -> TextEdit | None:
    """Un seul remplacement couvrant les lignes qui changent (préfixe et suffixe communs exclus)."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    if a == b:
        return None
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a, end_b = end_a - 1, end_b - 1
    return TextEdit(start, end_a, "".join(b[start:end_b]))
