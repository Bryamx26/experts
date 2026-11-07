# Experts

Compagnon de bureau local qui aide à coder et configurer, avec un spécialiste par
technologie : Python (le chat), Docker (la boîte), Kubernetes (la pieuvre).
Tout tourne en local avec Ollama, sans API cloud.

## État : étapes 1 et 2 de la feuille de route

Un compagnon Python en terminal qui appelle Ollama et utilise des outils :

| Outil | Rôle | Garde-fou |
|---|---|---|
| `read_file` | lit un fichier avec numéros de ligne | confiné à l'espace de travail (liens symboliques compris), 200 Ko max |
| `list_files` | liste un dossier | confiné à l'espace de travail |
| `validate_file` | ruff, hadolint, yamllint, kubeconform, docker compose | commandes fixes, lecture seule, erreurs jamais inventées par le LLM |
| `search_docs` | cherche dans la doc indexée (ChromaDB) | collections python, docker, kubernetes, tldr |
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

### Validateurs (étape 2)

Chaque validateur absent est signalé sans bloquer le reste.

```bash
pip install ruff yamllint                     # Python et YAML
# hadolint : https://github.com/hadolint/hadolint/releases (binaire unique)
# kubeconform : https://github.com/yannh/kubeconform/releases (binaire unique)
# docker compose : fourni avec Docker
```

Les erreurs pointent la ligne exacte, même pour kubeconform et docker compose
qui ne donnent qu'un chemin (`/spec/replicas`) : le chemin est retrouvé dans le
YAML. Les schémas Kubernetes sont mis en cache dans `~/.cache/experts/kubeconform`
(hors ligne après le premier passage). `validate_content()` sert à revalider une
correction proposée par le LLM avant de l'afficher.

### Documentation indexée (étape 2)

```bash
docker exec ollama ollama pull nomic-embed-text   # modèle d'embeddings local
mkdir -p ~/docs && cd ~/docs
git clone --depth 1 https://github.com/docker/docs docker-docs
git clone --depth 1 https://github.com/kubernetes/website k8s-website
git clone --depth 1 https://github.com/tldr-pages/tldr
# Python : archive « Plain text » sur https://docs.python.org/3/download.html, à décompresser

python -m experts.index docker ~/docs/docker-docs/content
python -m experts.index kubernetes ~/docs/k8s-website/content/en/docs
python -m experts.index tldr ~/docs/tldr/pages/common
python -m experts.index python ~/docs/python-3-docs-text
```

La base est stockée dans `~/.local/share/experts/chroma` (`EXPERTS_DATA_DIR`).
Réindexer un dossier remplace ses morceaux au lieu de les dupliquer.

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

Variables d'environnement : `EXPERTS_EMBED_MODEL`, `EXPERTS_DATA_DIR`, `EXPERTS_BACKEND` (`ollama` ou `anthropic`), `EXPERTS_CLOUD_MODEL`, `EXPERTS_OLLAMA_HOST`, `EXPERTS_CHAT_MODEL`,
`EXPERTS_SEARXNG_URL`, `EXPERTS_WORKSPACE`, `EXPERTS_TIMEOUT`.

## Vérifications

```bash
pytest          # sans Ollama ni SearXNG ; tests de validateurs ignorés si l'outil manque
ruff check . && ruff format --check .
mypy experts tests
```

## Structure

```
experts/
├── companions/   # un YAML par spécialiste (étape 3)
├── assets/       # sprites : python/chat.svg, docker/boite.svg, kubernetes/pieuvre.svg
├── core/         # config.py, llm.py, agent.py, cloud.py, rag.py ; router.py à l'étape 3
├── lsp/          # serveur pygls (étape 4)
├── tools/        # base.py, files.py, web_search.py, shell.py, validators.py
└── ui/           # fenêtre PySide6 (étape 5)
```

## Feuille de route

1. ✅ Compagnon en terminal qui appelle Ollama avec ses outils.
2. ✅ RAG et validateurs.
3. YAML de compagnons et routeur.
4. Serveur LSP : diagnostics, puis code actions.
5. Interface PySide6 et liaison WebSocket.
6. Autocomplétion FIM.
