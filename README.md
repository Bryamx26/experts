# Experts

Compagnon de bureau local qui aide à coder et configurer, avec un spécialiste par
technologie : Python (le chat), Docker (la boîte), Kubernetes (la pieuvre).
Tout tourne en local avec Ollama, sans API cloud.

## État : étape 1 de la feuille de route

Un compagnon Python en terminal qui appelle Ollama et utilise des outils :

| Outil | Rôle | Garde-fou |
|---|---|---|
| `read_file` | lit un fichier avec numéros de ligne | confiné à l'espace de travail (liens symboliques compris), 200 Ko max |
| `list_files` | liste un dossier | confiné à l'espace de travail |
| `web_search` | recherche via SearXNG local | résultats traités comme des données, pas des consignes |
| `run_command` | exécute une commande | **confirmation à chaque appel**, sans shell, délai de 60 s |

## Démarrage

```bash
# 1. Ollama, de préférence isolé en conteneur et exposé sur localhost seulement
docker run -d --name ollama -p 127.0.0.1:11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec ollama ollama pull qwen3.5:4b

# 2. (Optionnel) SearXNG local, avec `formats: [html, json]` dans settings.yml
docker run -d --name searxng -p 127.0.0.1:8888:8080 searxng/searxng

# 3. Le compagnon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m experts --workspace ~/mon-projet
```

### Option : modèle payant (désactivée par défaut)

Le projet reste local par défaut. Pour une question difficile, tu peux passer
sur Claude (API Anthropic, payante) :

```bash
pip install -e ".[cloud]"
export ANTHROPIC_API_KEY=...   # jamais dans le code ni dans Git
python -m experts --cloud --workspace ~/mon-projet
```

Le compagnon affiche un avertissement et demande confirmation : en mode cloud,
tes questions et le contenu des fichiers lus partent chez Anthropic. Les
fichiers de secrets (`.env`, `*.key`, `*.pem`, clés SSH…) sont refusés par
`read_file` dans tous les modes. Modèle par défaut : `claude-opus-5-5`
(`EXPERTS_CLOUD_MODEL` pour changer).

Variables d'environnement : `EXPERTS_BACKEND` (`ollama` ou `anthropic`), `EXPERTS_CLOUD_MODEL`, `EXPERTS_OLLAMA_HOST`, `EXPERTS_CHAT_MODEL`,
`EXPERTS_SEARXNG_URL`, `EXPERTS_WORKSPACE`, `EXPERTS_TIMEOUT`.

## Vérifications

```bash
pytest          # tests sans Ollama ni SearXNG (faux transports)
ruff check . && ruff format --check .
mypy experts tests
```

## Structure

```
experts/
├── companions/   # un YAML par spécialiste (étape 3)
├── assets/       # sprites : python/chat.svg, docker/boite.svg, kubernetes/pieuvre.svg
├── core/         # config.py, llm.py, agent.py ; rag.py et router.py aux étapes 2-3
├── lsp/          # serveur pygls (étape 4)
├── tools/        # base.py, files.py, web_search.py, shell.py ; validators.py à l'étape 2
└── ui/           # fenêtre PySide6 (étape 5)
```

## Feuille de route

1. ✅ Compagnon en terminal qui appelle Ollama avec ses outils.
2. RAG et validateurs.
3. YAML de compagnons et routeur.
4. Serveur LSP : diagnostics, puis code actions.
5. Interface PySide6 et liaison WebSocket.
6. Autocomplétion FIM.
