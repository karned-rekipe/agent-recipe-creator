# API et MCP

L'API REST publique de préproduction est exposée sous `/api/recipe-agent`.
`POST /v1/recipes/ai-create` accepte une entrée texte, URL ou document base64,
un indicateur de génération d'image et un identifiant de suivi. La source est
exclusive afin d'éviter les interprétations ambiguës.

`POST /v1/recipes/{uuid}/ai-enrich` met à niveau une fiche existante sans
écraser les favoris, tags, images et métadonnées humaines déjà valides. Les
routes `agent-runs` permettent au frontend de reprendre un traitement en cours
après navigation ou rafraîchissement.

Le MCP interne expose `ai_create_recipe` et `ai_enrich_recipe`. Il n'est pas
publié par un Ingress tant qu'une protection équivalente à celle du MCP Recipe
n'est pas disponible.
