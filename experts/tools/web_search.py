"""Recherche web via une instance SearXNG auto-hébergée (API JSON).

Prérequis : activer le format json dans settings.yml de SearXNG
(search.formats: [html, json]).
Les résultats sont des données non fiables : le prompt système le rappelle au
modèle, et aucune commande ne part sans confirmation de toute façon.
"""

from __future__ import annotations

import httpx

from experts.tools.base import Tool, ToolError


def make_web_search(
    searxng_url: str,
    *,
    max_results: int = 5,
    transport: httpx.BaseTransport | None = None,
) -> Tool:
    client = httpx.Client(base_url=searxng_url, timeout=15.0, transport=transport)

    def web_search(query: str) -> str:
        try:
            response = client.get("/search", params={"q": query, "format": "json"})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"SearXNG indisponible sur {searxng_url} ({exc}).") from exc
        results = response.json().get("results", [])[:max_results]
        if not results:
            return "Aucun résultat."
        return "\n\n".join(
            f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}"
            for i, r in enumerate(results, start=1)
        )

    return Tool(
        name="web_search",
        description="Recherche sur le web. Renvoie titres, URL et extraits à citer.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Requête de recherche."}},
            "required": ["query"],
        },
        func=web_search,
    )
