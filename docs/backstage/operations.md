# Exploitation

Surveiller les probes, les logs structurés et les `AgentRun` bloqués en statut
`running`. Les champs `metadata.current_stage`, `stage_history` et
`tracking_id` expliquent où se trouve un traitement et permettent à l'UI de le
reprendre.

Un échec OCR ne doit pas persister le document. Un échec de génération d'image
ne doit pas annuler une recette correctement structurée : l'avertissement est
retourné et la fiche reste modifiable.

Pour un diagnostic, vérifier dans l'ordre Recipe, OCR, le fournisseur LLM puis
MongoDB AgentRun. Un rollback consiste à restaurer le digest précédent de
l'agent dans GitOps ; il n'implique aucune migration des recettes.

Un `401 invalid_token` sur le MCP Recipe indique que le client credentials de
l'agent n'est pas présent, que son secret Vault est désynchronisé, ou que le
jeton ne contient pas l'audience, le rôle et les scopes attendus. Contrôler le
client Keycloak `rekipe-agent-recipe-creator`, l'ExternalSecret du runtime et
la configuration `mcp_registry.oauth` sans afficher le secret ni le jeton.
