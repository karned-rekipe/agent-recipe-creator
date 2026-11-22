# Architecture

Le pipeline LangGraph sépare la planification, la revue, la résolution des
catalogues, la persistance et la génération d'image. Les nœuds utilisent le MCP
Recipe via un adapter, jamais un accès direct à MongoDB métier.

Le résolveur d'entrée accepte exactement une source : texte, URL ou document.
Une page Web est analysée avec Schema.org puis un repli textuel protégé contre
SSRF. Une image ou un PDF est envoyé au MCP OCR. Le document et le texte OCR ne
sont pas persistés par `ocr`.

Les créations et enrichissements convergent vers le même `RecipePlan`. Le
second passage de revue complète les métadonnées manquantes et recalcule les
temps à partir des étapes avant l'écriture finale.

Avant la planification, l'agent charge les tags existants depuis le MCP Recipe.
Le plan choisit uniquement dans ce catalogue, puis un nœud de résolution
rejette toute valeur inconnue et fusionne les UUID sélectionnés avec les tags
humains de la fiche.
