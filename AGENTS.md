# AGENTS.md — `agent-recipe-creator/`

Document d'entrée pour tout agent de codage qui intervient sur ce repo.
Lecture obligatoire avant toute modification.

Pour le contexte global du projet et les conventions transverses, voir `../AGENTS.md` à la racine du workspace.

---

## Contexte global

`agent-recipe-creator/` est l'agent IA de Rekipe. Il transforme du texte brut,
une fiche Web ou le texte OCR d'une image en entités structurées persistées
dans `recipe/`. Il peut aussi mettre à niveau une recette existante. Il ne
possède aucune entité métier recette propre — il délègue toute persistance au
service `recipe/`.

## Rôle

- Structurer une recette en langage naturel via un LLM (`_PydanticAIPlanner`)
- Résoudre une entrée texte, URL de fiche, image ou URL d'image
- Compléter origine, difficulté et prix ; ventiler les temps dans les étapes
- Résoudre les ingrédients et ustensiles existants par fuzzy matching (`rapidfuzz`)
- Créer les entités manquantes et créer la recette complète dans `recipe/`
- Reclasser la description d'une fiche existante vers ingrédients et étapes
- Déclencher la génération de l'image principale via le tool MCP de `recipe/`
- Persister un `AgentRun` (traçabilité de chaque exécution)
- Exposer le résultat via REST (`:8006`), MCP (`:8004`) et CLI

## Règles de développement

- Architecture hexagonale : `domain → application → adapters → infrastructure`
- `domain/` n'importe que `arclith` (Entity) et `pydantic` — zéro dépendance framework
- Toute interaction avec `recipe/` passe par `_MCPRecipeRegistry` (adapter output)
- Toute extraction d'image passe par `MCPOCRClient` vers le MCP de `ocr/`
- Le graphe LangGraph est construit et compilé dans `agent_adapter.py` uniquement
- Les nœuds du graphe sont des factories dans `_nodes.py` — ne pas instancier des classes de nœuds
- Ports : REST `:8006`, MCP `:8004`
- Tests ≥ 80 % — `make coverage`
- Pre-commit : `make precommit` (lint + typecheck + security)

## Architecture locale

```
domain/
  models/
    agent_run.py    # AgentRun — entité de traçabilité (status, recipe_uuid, metadata)
    recipe.py       # RecipePlan, RecipeResult, IngredientLine, UstensilLine, RecipeStep
    recipe_intake.py # Entrée exclusive texte, URL, image ou URL d'image
    ingredient.py   # Ingredient local (non persisté)
  ports/
    recipe_agent.py # RecipeAgentPort — création et enrichissement en place
    recipe_ocr.py   # RecipeOCRPort — extraction stateless d'une image
    recipe_source_loader.py # RecipeSourceLoaderPort — chargement Web public

application/
  use_cases/
    process_raw_recipe.py  # Création multisource, traçabilité et appel agent
    enrich_existing_recipe.py # Mise à niveau d'une recette existante
    find_by_name.py        # FindByNameUseCase — recherche d AgentRun par nom
  services/
    recipe_service.py      # Façade ai_create() et ai_enrich()
    recipe_input_resolver.py # Dispatch texte/Web/OCR
    agent_run_service.py   # AgentRunService — CRUD AgentRun

adapters/
  input/
    cli/recipe_cli.py          # Typer CLI — commande create
    fastapi/                   # Routes REST :8006
    fastmcp/                   # Tools MCP :8004
    schemas/                   # Schemas de serialisation
  output/
    mongodb/agent_run_repository.py    # MongoDBAgentRunRepository
    memory/                            # InMemoryAgentRunRepository (tests)
    recipe_agent/
      agent_adapter.py   # RecipeAgentAdapter — orchestre le graphe LangGraph
      _state.py          # RecipeAgentState (TypedDict LangGraph)
      _nodes.py          # Tous les noeuds (factories make_*_node)
      _planner.py        # _PydanticAIPlanner — appel LLM structuré
      _mcp_registry.py   # _MCPRecipeRegistry — client MCP vers recipe/
      _fuzzy.py          # make_fuzzy_matcher(threshold)
      _logger.py         # logger module-level
    ocr/mcp_ocr_client.py               # client MCP vers ocr/
    source_loader/web_recipe_source_loader.py # Schema.org + repli texte, anti-SSRF

infrastructure/
  config.py      # AgentConfig — mcp_registry, recipe_api, lm, fuzzy (Pydantic)
  container.py   # build_container() — DI : repo + agent + use case + service
  ingredient_container.py
```

## Graphes LangGraph

```
Création :
[plan] → [review_plan] → [check_recipe]
                            ├─ exists → END
                            └─ new → [create_components]
                                      → [resolve_ingredients]
                                      → [resolve_ustensils]
                                      → [create_recipe]
                                      → [generate_image] → END

Mise à niveau :
[plan] → [review_plan] → [create_components]
                       → [resolve_ingredients]
                       → [resolve_ustensils]
                       → [update_recipe]
                       → [generate_image] → END
```

| Nœud                  | Rôle                                                        | MCP tools appelés                       |
|-----------------------|-------------------------------------------------------------|-----------------------------------------|
| `plan`                | LLM extrait `RecipePlan` depuis le texte brut               | — (pydantic-ai)                         |
| `review_plan`         | Second passage de cohérence et complétion                   | — (pydantic-ai)                         |
| `check_recipe`        | Fuzzy match sur `list_recipes(name)`                        | `list_recipes`                          |
| `create_components`   | Résout les éventuelles sous-recettes                        | tools recettes                          |
| `resolve_ingredients` | Pour chaque ingredient : fuzzy match ou `create_ingredient` | `list_ingredients`, `create_ingredient` |
| `resolve_ustensils`   | Pour chaque ustensile : fuzzy match ou `create_equipment`   | `list_equipment`, `create_equipment`    |
| `create_recipe`       | Crée la recette complète avec références UUID               | `create_recipe`                         |
| `update_recipe`       | Reclasse et complète une recette existante en place         | `update_recipe`                         |
| `generate_image`      | Génère l'image si elle est absente et demandée              | `generate_recipe_image`                 |

## State LangGraph (`RecipeAgentState`)

| Clé                    | Type                 | Contenu                                    |
|------------------------|----------------------|--------------------------------------------|
| `messages`             | `list[BaseMessage]`  | Historique LangChain (avec `add_messages`) |
| `plan`                 | `RecipePlan \| None` | Sortie du nœud `plan`                      |
| `resolved_ingredients` | `dict[str, str]`     | `nom → uuid` des ingrédients résolus       |
| `resolved_ustensils`   | `dict[str, str]`     | `nom → uuid` des ustensiles résolus        |
| `recipe_uuid`          | `str \| None`        | UUID de la recette créée                   |
| `recipe_exists`        | `bool \| None`       | True si recette déjà existante             |
| `existing_recipe`      | `dict \| None`        | Fiche hydratée à mettre à niveau            |
| `input_sources`        | `list[RecipeSource]`   | Provenance Web ou utilisateur               |
| `generate_image`       | `bool`                 | Demande de génération d'image               |
| `warnings`             | `list[str]`            | Avertissements non bloquants                 |
| `error`                | `str \| None`        | Erreur éventuelle                          |

## Configuration (`config.yaml`)

```yaml
mcp_registry:
  url: http://127.0.0.1:8302/mcp    # URL du serveur MCP recipe/

ocr:
  url: http://127.0.0.1:8041/mcp    # URL du serveur MCP ocr/
  timeout_seconds: 120

recipe_api:
  url: http://127.0.0.1:8301        # URL HTTP du service recipe/
  tenant_uri: foyer-demo

lm:
  planner:
    provider: openai                 # anthropic | openai
    model_name: gpt-5.4-mini
    api_key: ${OPENAI_API_KEY}       # fallback ; la valeur cible vient de Vault via config/secrets.yaml
    # Pour un LLM local (LM Studio / Ollama) :
    # provider: openai
    # base_url: http://127.0.0.1:1234/v1
    # api_key: lm-studio

fuzzy:
  threshold: 80   # Score minimum (0-100) pour un match fuzzy

adapters:
  repository: mongodb
  mongodb:
    uri: mongodb://localhost:5971
    db_name: agent-recipe-creator

api:
  port: 8006
mcp:
  port: 8004
```

## Ordre de démarrage (dev local)

```bash
# Depuis la racine du workspace : charge .env.dev, provisionne Vault et lance
# recipe, les agents recette/ingrédients, media, OCR, le frontend et les ponts.
./rekipe_dev.sh up recipe-stack

curl -fsS http://127.0.0.1:8006/openapi.json >/dev/null
curl -fsS http://127.0.0.1:8040/ready
```

## Commandes utiles

```bash
make setup      # git config core.hooksPath .githooks
make precommit  # lint + typecheck + security
make quality    # lint + security + complexity + typecheck + coverage
make test       # pytest -v
make test-unit  # pytest -v -m "not e2e"

# CLI direct
uv run --frozen python main_cli.py "Recette de carbonara"
```

## Fichiers à lire en premier

1. `adapters/output/recipe_agent/agent_adapter.py` — graphe LangGraph complet
2. `adapters/output/recipe_agent/_state.py` — structure du state
3. `adapters/output/recipe_agent/_nodes.py` — tous les nœuds
4. `adapters/output/recipe_agent/_mcp_registry.py` — appels MCP vers recipe/
5. `application/services/recipe_input_resolver.py` — dispatch texte/Web/OCR
6. `application/use_cases/process_raw_recipe.py` — création + traçabilité
7. `application/use_cases/enrich_existing_recipe.py` — mise à niveau en place
8. `infrastructure/config.py` — AgentConfig (MCP, OCR, LLM, fuzzy)
9. `infrastructure/container.py` — DI et câblage

## Règle du journal — rappel

Toute modification non triviale doit être consignée dans `journal/YYYY-MM-DD-slug.md` à la **racine du workspace**. Voir `../AGENTS.md` §6 et `../journal/README.md`.

Cas spécifiques à ce repo qui exigent une entrée journal :

- ajout / retrait / renommage d'un nœud du graphe LangGraph ;
- changement du contrat MCP avec `recipe/` (nouveau tool consommé, signature différente) ;
- changement de provider ou de modèle LLM par défaut ;
- ajustement structurant du seuil de fuzzy matching ou de la stratégie de résolution ;
- changement de port (REST `8006`, MCP `8004`).

## Hors scope

- La **persistance des recettes** : ce repo ne persiste rien d'autre que `AgentRun`. Tout le reste passe par `recipe/` via MCP.
- Le **catalogue ingrédients / équipements / tags** : il vit dans `recipe/`. Cet agent y crée ou y résout des entrées via les tools MCP exposés par `recipe/`, jamais via accès direct à la base.
- Le **frontend** : `gwel/`.
