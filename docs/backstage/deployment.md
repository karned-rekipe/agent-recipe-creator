# Déploiement

L'image `ghcr.io/karned-rekipe/agent-recipe-creator` contient les exécutables
REST et MCP. GitOps déploie deux processus séparés dans le namespace `rekipe` :
API sur `8006` et MCP interne sur `8004`.

La configuration injecte les URLs internes de Recipe et OCR, le tenant
`rekipe-preprod`, l'URI MongoDB des AgentRun et la clé du fournisseur LLM. Les
secrets viennent de Vault via External Secrets. Le MCP OCR doit être prêt avant
les imports de documents ; les entrées texte et Web restent indépendantes.

Les egress sont limités à MongoDB, Recipe, OCR et HTTPS vers le fournisseur.
