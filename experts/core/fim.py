"""Complétion « fill-in-the-middle » (texte grisé) avec un petit modèle de code.

Choix :
- Modèle dédié et petit (Qwen2.5-Coder 1.5B par défaut) : la complétion doit
  arriver en quelques centaines de ms, le modèle 4B reste pour les explications ;
- API /api/generate d'Ollama avec `suffix` : Ollama applique le gabarit FIM du
  modèle (<|fim_prefix|>…), on ne le code pas en dur ;
- contexte borné (3 000 caractères avant, 1 000 après) : la latence dépend surtout
  de la longueur du prompt ;
- client asynchrone : le serveur LSP n'est jamais bloqué, et une requête
  annulée par l'éditeur ne laisse rien tourner.
"""

from __future__ import annotations

import httpx

PREFIX_CHARS = 3_000
SUFFIX_CHARS = 1_000
MAX_LINES = 8


def clean_completion(text: str, suffix: str) -> str:
    """Nettoie la proposition : longueur bornée, pas de répétition du texte qui suit."""
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")[:MAX_LINES]
    text = "\n".join(lines).rstrip()
    following = suffix.lstrip("\n").split("\n", 1)[0].strip()
    if following and following in text:
        text = text[: text.index(following)].rstrip()
    return text if text.strip() else ""


class FimClient:
    def __init__(
        self,
        host: str,
        model: str = "qwen2.5-coder:1.5b",
        *,
        timeout: float = 10.0,
        max_tokens: int = 96,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._http = httpx.AsyncClient(base_url=host, timeout=timeout, transport=transport)

    async def complete(self, prefix: str, suffix: str) -> str:
        prefix, suffix = prefix[-PREFIX_CHARS:], suffix[:SUFFIX_CHARS]
        body = {
            "model": self.model,
            "prompt": prefix,
            "suffix": suffix,
            "stream": False,
            "options": {"num_predict": self._max_tokens, "temperature": 0.2, "stop": ["\n\n\n"]},
        }
        try:
            r = await self._http.post("/api/generate", json=body)
            r.raise_for_status()
        except httpx.HTTPError:
            return ""  # pas de complétion plutôt qu'une erreur à chaque frappe
        return clean_completion(str(r.json().get("response", "")), suffix)

    async def aclose(self) -> None:
        await self._http.aclose()
