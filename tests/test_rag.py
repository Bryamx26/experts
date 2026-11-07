"""RAG : vrai ChromaDB sur disque temporaire, faux embedder déterministe."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import httpx

from experts.core.rag import DocStore, OllamaEmbedder, make_search_docs, split_text
from experts.tools.base import ToolRegistry

VOCAB = ["dockerfile", "expose", "port", "pod", "replicas", "deployment", "python", "list", "healthcheck"]


class BagOfWords:
    """Embedder de test : un vecteur de comptage sur un petit vocabulaire."""

    model = "test-bow"

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"\w+", text.lower())
        vec = [float(words.count(w)) for w in VOCAB] + [0.01]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def test_split_by_headings_and_size() -> None:
    text = "# Intro\nBonjour.\n\n## Ports\n" + ("EXPOSE 80 ouvre un port.\n\n" * 120)
    chunks = split_text(text, "docker.md")
    assert chunks[0].heading == "Intro"
    assert all(c.heading == "Ports" for c in chunks[1:])
    assert len(chunks) > 2 and all(len(c.text) <= 1_200 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_index_and_search(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "expose.md").write_text("# EXPOSE\nThe EXPOSE instruction documents the port of a Dockerfile.\n")
    (docs / "healthcheck.md").write_text("# HEALTHCHECK\nHEALTHCHECK tells Docker how to test a container.\n")
    (docs / ".cache.md").write_text("# caché\nport port port\n")
    store = DocStore(tmp_path / "chroma", BagOfWords())

    assert store.index_directory("docker", docs) == 2
    assert store.index_directory("docker", docs) == 2  # réindexer ne duplique pas
    assert store.count("docker") == 2
    hits = store.search("docker", "which port does my dockerfile expose")
    assert hits[0].source == "expose.md"


def test_search_docs_tool(tmp_path: Path) -> None:
    store = DocStore(tmp_path / "chroma", BagOfWords())
    registry = ToolRegistry([make_search_docs(store)])
    assert "Rien d'indexé" in registry.call("search_docs", {"query": "pod", "collection": "kubernetes"})
    assert "Collection inconnue" in registry.call("search_docs", {"query": "x", "collection": "rust"})


def test_ollama_embedder_uses_nomic_prefixes() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})

    emb = OllamaEmbedder("http://ollama.test", transport=httpx.MockTransport(handler))
    assert emb.embed_query("pod") == [0.1, 0.2]
    emb.embed_documents(["a", "b"])
    assert seen[0] == {"model": "nomic-embed-text", "input": ["search_query: pod"]}
    assert seen[1]["input"] == ["search_document: a", "search_document: b"]
