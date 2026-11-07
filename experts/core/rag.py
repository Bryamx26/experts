"""RAG local : ChromaDB, une collection par spécialiste, embeddings via Ollama.

Choix :
- Embeddings calculés par Ollama (`nomic-embed-text`) plutôt que par le modèle
  ONNX par défaut de Chroma : un seul service local à gérer, rien n'est
  téléchargé en douce au premier appel, et le modèle tourne sur le GPU s'il y en a un.
- On passe les vecteurs nous-mêmes à Chroma (`embeddings=`, `query_embeddings=`),
  ce qui rend le code testable avec un faux embedder.
- Découpage par titres Markdown, puis par blocs d'environ 1 200 caractères avec
  chevauchement : assez court pour le contexte d'un modèle 4B, assez long pour
  garder un exemple de code entier.
- Identifiants déterministes (chemin + numéro de morceau) : réindexer remplace
  au lieu de dupliquer.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import httpx

from experts.core.llm import LLMError
from experts.tools.base import Tool, ToolError

COLLECTIONS = ("python", "docker", "kubernetes", "tldr")
DOC_SUFFIXES = (".md", ".markdown", ".rst", ".txt")
CHUNK_CHARS = 1_200
OVERLAP_CHARS = 150
_HEADING_RE = re.compile(r"^(#{1,4} .+|.+\n[=\-~^]{3,})$", re.MULTILINE)


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(
        self,
        host: str,
        model: str = "nomic-embed-text",
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._http = httpx.Client(base_url=host, timeout=timeout, transport=transport)
        # nomic-embed-text est entraîné avec ces préfixes ; sans eux la recherche est moins bonne.
        self._nomic = model.startswith("nomic-embed")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            r = self._http.post("/api/embed", json={"model": self.model, "input": texts})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Embeddings Ollama indisponibles ({exc}). `ollama pull {self.model}` ?") from exc
        embeddings: list[list[float]] = r.json()["embeddings"]
        return embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefix = "search_document: " if self._nomic else ""
        return self._embed([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        prefix = "search_query: " if self._nomic else ""
        return self._embed([prefix + text])[0]


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    source: str
    heading: str


@dataclass(frozen=True, slots=True)
class Hit:
    text: str
    source: str
    heading: str
    distance: float


def split_text(text: str, source: str) -> list[Chunk]:
    """Découpe par sections (titres Markdown ou RST), puis par taille."""
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    bounds = [0, *[s for s in starts if s > 0], len(text)]
    chunks: list[Chunk] = []
    for a, b in pairwise(bounds):
        section = text[a:b].strip()
        if not section:
            continue
        heading = section.splitlines()[0].lstrip("# ").strip()[:120]
        pos = 0
        while pos < len(section):
            end = min(pos + CHUNK_CHARS, len(section))
            if end < len(section):  # coupe de préférence sur une ligne vide
                cut = section.rfind("\n\n", pos + CHUNK_CHARS // 2, end)
                end = cut if cut != -1 else end
            piece = section[pos:end].strip()
            if piece:
                digest = hashlib.sha1(f"{source}:{len(chunks)}".encode()).hexdigest()[:16]
                chunks.append(Chunk(id=digest, text=piece, source=source, heading=heading))
            if end >= len(section):
                break
            pos = max(end - OVERLAP_CHARS, pos + 1)
    return chunks


def iter_doc_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in DOC_SUFFIXES
            and not any(p.startswith(".") for p in path.parts)
        ):
            yield path


class DocStore:
    def __init__(self, data_dir: Path, embedder: Embedder, *, client: Any = None) -> None:
        if client is None:
            import chromadb

            data_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(data_dir))
        self._client = client
        self._embedder = embedder

    def _collection(self, name: str) -> Any:
        if name not in COLLECTIONS:
            raise ToolError(f"Collection inconnue '{name}'. Choix : {', '.join(COLLECTIONS)}.")
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embed_model": self._embedder.model},
        )

    def index_directory(self, collection: str, root: Path, *, batch: int = 64) -> int:
        """Indexe tous les fichiers de doc sous `root`. Renvoie le nombre de morceaux."""
        coll = self._collection(collection)
        chunks = [
            c
            for f in iter_doc_files(root)
            for c in split_text(f.read_text(encoding="utf-8", errors="replace"), str(f.relative_to(root)))
        ]
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            coll.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=self._embedder.embed_documents([c.text for c in part]),
                metadatas=[{"source": c.source, "heading": c.heading} for c in part],
            )
        return len(chunks)

    def search(self, collection: str, query: str, k: int = 4) -> list[Hit]:
        coll = self._collection(collection)
        if coll.count() == 0:
            return []
        res = coll.query(query_embeddings=[self._embedder.embed_query(query)], n_results=k)
        return [
            Hit(text=doc, source=meta["source"], heading=meta["heading"], distance=dist)
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def count(self, collection: str) -> int:
        return int(self._collection(collection).count())


def make_search_docs(store: DocStore) -> Tool:
    def search_docs(query: str, collection: str) -> str:
        try:
            hits = store.search(collection, query)
        except LLMError as exc:
            raise ToolError(str(exc)) from exc
        if not hits:
            return f"Rien d'indexé dans '{collection}' pour cette requête. Lance `python -m experts.index`."
        return "\n\n".join(f"[{h.source} — {h.heading}]\n{h.text}" for h in hits)

    return Tool(
        name="search_docs",
        description="Cherche dans la documentation indexée localement. Cite la source renvoyée.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question ou mots-clés."},
                "collection": {"type": "string", "enum": list(COLLECTIONS)},
            },
            "required": ["query", "collection"],
        },
        func=search_docs,
    )
