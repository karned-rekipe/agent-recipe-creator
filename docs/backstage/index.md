# Agent de création de recettes

L'agent transforme une source texte, une page Web, une image ou un PDF en
recette structurée. Il peut également compléter une fiche existante : origine,
difficulté, prix, saison, temps par étape, ingrédients et instructions mal
classés, puis image principale lorsqu'elle manque.

La persistance métier est exclusivement déléguée à `recipe`. Le service ne
duplique pas les entités recette ; il conserve uniquement les `AgentRun` qui
tracent le statut, la progression, les avertissements et le résultat.

Les traitements durent couramment plusieurs dizaines de secondes. Les phases
réelles du graphe sont exposées à l'interface afin d'afficher une progression
plutôt qu'un simple état bloquant.
