# Rekipe Recipe Creator Agent

Agent IA de création et de mise à niveau des recettes. Il accepte trois
parcours utilisateur tout en conservant les quatre champs historiques de son
contrat :

- texte copié-collé ;
- URL publique d'une fiche recette, d'une image ou d'un PDF ;
- document image ou PDF envoyé en base64.

Les pages Web sont d'abord lues via leurs données Schema.org `Recipe`, avec un
repli sur le texte visible. Les images et PDF passent par le microservice `ocr` via
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

Lorsque le MCP Recipe est protégé, l'agent obtient un jeton court par le flux
OAuth `client_credentials`. Son client Keycloak ne reçoit que les scopes
`recipes:read`, `recipes:write` et `recipes:image:generate`; le secret reste
dans Vault et n'est jamais inscrit dans la configuration versionnée.

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
- progression d'une création : `GET http://127.0.0.1:8006/v1/agent-runs/creations/{tracking_id}/running` ;
- progression d'une mise à niveau : `GET http://127.0.0.1:8006/v1/agent-runs/recipes/{uuid}/running` ;
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

Le champ optionnel `tracking_id` associe une requête de création à son
`AgentRun`. Il doit être unique côté client pour permettre le suivi de la phase
courante pendant l'appel synchrone.

## MCP

Le serveur MCP HTTP de l'agent écoute sur `http://127.0.0.1:8004/mcp` et expose
notamment :

- `ai_create_recipe(raw_text, allow_duplicate=false)` pour créer une recette ;
- `ai_enrich_recipe(recipe_uuid, generate_image=true)` pour compléter et
  reclasser une fiche existante en place.

Le second tool renvoie l'UUID et le nom de la recette, les indicateurs
`updated` et `image_generated`, ainsi que les éventuels avertissements.

Pendant une création ou une mise à niveau, l'`AgentRun` publie
`metadata.current_stage` et `metadata.stage_history`. Les phases correspondent
aux nœuds réellement exécutés : lecture, analyse, vérification, contrôle des
doublons pour la création, organisation, résolution des ingrédients,
ustensiles et tags existants, sauvegarde puis traitement de l'image.

Le catalogue de tags est lu via le MCP Recipe avant l'analyse. Le LLM ne peut
proposer que les noms fournis ; une seconde résolution déterministe ignore les
valeurs inconnues. Pour une fiche existante, les tags détectés sont fusionnés
avec les UUID déjà associés et ne les remplacent jamais.

## Validation

```bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen mypy domain application adapters infrastructure
uv run --frozen pytest --cov=. --cov-fail-under=80
```
