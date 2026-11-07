"""Indexation de documentation dans ChromaDB.

Exemple : python -m experts.index docker ~/docs/docker-docs/content
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experts.core.config import Settings
from experts.core.llm import LLMError
from experts.core.rag import COLLECTIONS, DocStore, OllamaEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexe un dossier de documentation.")
    parser.add_argument("collection", choices=COLLECTIONS)
    parser.add_argument("folder", type=Path, help="Dossier contenant des .md, .rst ou .txt.")
    args = parser.parse_args()

    settings = Settings.from_env()
    store = DocStore(settings.data_dir / "chroma", OllamaEmbedder(settings.ollama_host, settings.embed_model))
    try:
        n = store.index_directory(args.collection, args.folder.expanduser().resolve())
    except LLMError as exc:
        raise SystemExit(f"[erreur] {exc}") from exc
    print(f"{n} morceaux indexés dans '{args.collection}' ({store.count(args.collection)} au total).")


if __name__ == "__main__":
    main()
