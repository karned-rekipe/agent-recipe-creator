# Rekipe Recipe Creator Agent

Agent IA de création et de mise à niveau des recettes. Il accepte quatre
sources exclusives :

- texte copié-collé ;
- URL publique d'une fiche recette ;
- image envoyée en base64 ;
- URL publique d'une image.

Les pages Web sont d'abord lues via leurs données Schema.org `Recipe`, avec un
repli sur le texte visible. Les images passent par le microservice `ocr` via
MCP. L'agent complète notamment l'origine, la difficulté, le prix et les temps
des étapes. La saison reste calculée par `recipe` depuis le catalogue
ingrédients.

Une fiche existante peut être mise à niveau en place : les ingrédients et
étapes présents dans la description sont reclassés, les métadonnées absentes
sont complétées et une image principale est générée si nécessaire. Les champs
humains déjà renseignés, les favoris, tags, images et métadonnées de migration
sont préservés.

## Architecture

Le repo suit l'architecture hexagonale `domain → application → adapters →
infrastructure`. La persistance des recettes et catalogues passe exclusivement
par le MCP de `recipe`; l'extraction d'image passe exclusivement par le MCP de
`ocr`. Seuls les `AgentRun` sont persistés localement par l'agent.

Consulter [AGENTS.md](./AGENTS.md) pour le graphe LangGraph, les contrats et les
règles de contribution.

## Démarrage local

Depuis la racine du workspace :

```bash
./rekipe_dev.sh up recipe-stack
curl -fsS http://127.0.0.1:8006/openapi.json >/dev/null
curl -fsS http://127.0.0.1:8040/ready
```

Le wrapper charge `.env.dev`, provisionne le secret OpenAI local dans Vault et
démarre les MCP nécessaires. Ne pas versionner de clé API.

## API REST

- création : `POST http://127.0.0.1:8006/v1/recipes/ai-create` ;
- mise à niveau : `POST http://127.0.0.1:8006/v1/recipes/{uuid}/ai-enrich` ;
- documentation OpenAPI : `http://127.0.0.1:8006/docs`.

Exemple texte sans génération d'image :

```bash
curl -X POST http://127.0.0.1:8006/v1/recipes/ai-create \
  -H 'Content-Type: application/json' \
  -d '{"raw_text":"Soupe : 4 carottes. Cuire 25 minutes puis mixer.","generate_image":false}'
```

Le contrat accepte exactement un champ parmi `raw_text`, `source_url`,
`image_url` et `image_base64`. Avec `image_base64`, `image_media_type` est
obligatoire.

## Validation

```bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen mypy domain application adapters infrastructure
uv run --frozen pytest --cov=. --cov-fail-under=80
```
