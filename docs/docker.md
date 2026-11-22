# Docker — agent-recipe-creator

## Structure de l'image

Build multistage 2 étapes :

| Stage     | Base                        | Rôle                                                    |
|-----------|-----------------------------|---------------------------------------------------------|
| `builder` | `python:3.13-slim-bookworm` | Installe les dépendances de prod dans `/app/.venv`      |
| `runtime` | `python:3.13-slim-bookworm` | Image finale — venv + code source, sans outils de build |

Même image de base pour les deux stages → un seul pull Docker Hub, pas de dépendance ghcr.io.

**Pourquoi `uv export` + `pip install` plutôt que `uv sync` ?**

uv est toujours utilisé, mais uniquement pour lire le lockfile :

1. `uv export --frozen --no-emit-project` → lit `uv.lock` et génère un `requirements.txt` avec versions et hashes
   fixés (pur PyPI, sans référence à des chemins locaux)
2. `pip install -r requirements.txt` → installe les packages depuis PyPI

`uv sync` tente de résoudre l'arborescence du projet (workspace, chemins relatifs) ce qui échoue en environnement isolé.
`uv export` court-circuite cette résolution et garantit la même reproductibilité que le lockfile.

L'image embarque la configuration de développement comme valeur de repli afin
de rester exécutable. En preproduction et en production, `config.yaml` est
remplacé par un ConfigMap monté en lecture seule ; les secrets sont injectés
séparément par l'environnement.

> **Prérequis :** le service `recipe` doit être démarré et accessible à l'URL configurée dans `mcp_registry.url`.

---

## Configuration locale

Les configs locales vivent dans `_configs/` à la racine des repos (gitignorées) :

```
_configs/
└── agent-recipe-creator/
    └── config.yaml   ← copié depuis agent-recipe-creator/config.yaml, à adapter
```

**Remplir obligatoirement dans `_configs/agent-recipe-creator/config.yaml`
ou fournir via Vault / variable d'environnement :**

- `lm.planner.api_key` — clé OpenAI (résolue par Vault via
  `config/secrets.yaml` ou injectée dans `OPENAI_API_KEY`)
- `mcp_registry.url` — URL MCP du service `recipe` joignable depuis le conteneur
- `mcp_registry.oauth` — en préproduction/production, URL du token Keycloak,
  identifiant du client confidentiel et scopes MCP minimaux ; le champ
  `client_secret` doit être résolu depuis Vault ou une variable d'environnement
- `recipe_api.url` — URL HTTP du service `recipe` joignable depuis le conteneur

Exemple pour un MCP Recipe protégé :

```yaml
mcp_registry:
  url: http://rekipe-recipe-mcp:8001/mcp
  oauth:
    token_url: http://keycloak/realms/rekipe/protocol/openid-connect/token
    client_id: rekipe-agent-recipe-creator
    client_secret: ${RECIPE_MCP_CLIENT_SECRET}
    scopes: [recipes:read, recipes:write, recipes:image:generate]
```

Chemin Vault attendu pour OpenAI :

```bash
vault kv put kv/rekipe/agent-recipe-creator/openai value="<cle-openai>"
```

---

## Build

> Depuis le dépôt `agent-recipe-creator`.

```bash
docker build -t agent-recipe-creator .
```

Rebuild propre après changement de dépendances (`pyproject.toml` / `uv.lock`) :

```bash
docker build --no-cache -t agent-recipe-creator .
```

---

## Lancement

> Depuis la racine des repos.

```bash
# Raccourci local pour le montage config
CONFIG="-v $(pwd)/_configs/agent-recipe-creator/config.yaml:/app/config.yaml:ro"

# Si config.yaml référence ${OPENAI_API_KEY}, la variable exportée sur l'hôte
# doit être transmise explicitement à Docker avec -e OPENAI_API_KEY.
export OPENAI_API_KEY="<cle-openai>"

# REST API  (port défini dans config.yaml → api.port)
docker run --rm $CONFIG -e OPENAI_API_KEY -p 8006:8006 agent-recipe-creator

# CLI interactif
docker run --rm -it $CONFIG -e OPENAI_API_KEY agent-recipe-creator python main_cli.py

# MCP SSE  (port défini dans config.yaml → mcp.port)
docker run --rm $CONFIG -e OPENAI_API_KEY -p 8004:8004 agent-recipe-creator python main_mcp_sse.py

# MCP Streamable HTTP
docker run --rm $CONFIG -e OPENAI_API_KEY -p 8004:8004 agent-recipe-creator python main_mcp_http.py

# MCP stdio
docker run --rm -i $CONFIG -e OPENAI_API_KEY agent-recipe-creator python main_mcp_stdio.py
```

On peut aussi remplacer `-e OPENAI_API_KEY` par
`--env-file _configs/agent-recipe-creator/agent.env` avec un fichier non
versionné contenant `OPENAI_API_KEY=...`.

---

## Production

En production, ne pas monter un fichier bind-mount contenant une clé API. Injecter la configuration non sensible via
ConfigMap et résoudre les secrets via Vault, ou utiliser un Docker/Kubernetes Secret si Vault n'est pas disponible.

**Docker Swarm / standalone secret :**

```bash
# Créer le secret (une seule fois)
docker secret create agent-config _configs/agent-recipe-creator/config.yaml

# Lancer avec le secret
docker run --rm \
  --mount type=secret,id=agent-config,target=/app/config.yaml \
  -p 8006:8006 \
  agent-recipe-creator
```

**Kubernetes ConfigMap :**

```yaml
# Le ConfigMap est monté via volumeMounts[].mountPath: /app/config.yaml
# La clé API est injectée via un Secret séparé (pas dans le ConfigMap)
```
