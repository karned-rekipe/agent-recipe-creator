from pydantic import AliasChoices, BaseModel, Field, computed_field


class IngredientLine(BaseModel):
    name: str = Field(description = "Nom de l'ingrédient (ex. Farine de blé)")
    unit: str | None = Field(default = None, description = "Unité de mesure (g, kg, ml, …). None si non applicable.")
    quantity: str | None = Field(default = None,
                                 description = "Quantité (ex. 180, 3-4, une pincée). None si non précisée.")


class UstensilLine(BaseModel):
    name: str = Field(description = "Nom de l'ustensile (ex. Fouet, Poêle)")


class RecipeStep(BaseModel):
    title: str = Field(description = "Titre court de l'étape (ex. Sabler la pâte)")
    instruction: str = Field(description = "Description complète de l'étape")
    preparation_minutes: int | None = Field(
        default = None,
        ge = 0,
        validation_alias=AliasChoices("preparation_minutes", "duration_minutes"),
        description = "Temps actif de préparation en minutes. None si non applicable ou inconnu.",
    )
    cooking_minutes: int | None = Field(
        default = None,
        ge = 0,
        description = "Temps de cuisson en minutes. None si non applicable ou inconnu.",
    )
    rest_minutes: int | None = Field(
        default = None,
        ge = 0,
        description = "Temps de repos, pousse, marinade ou refroidissement en minutes.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_minutes(self) -> int:
        return (self.preparation_minutes or 0) + (self.cooking_minutes or 0) + (self.rest_minutes or 0)

    @property
    def duration_minutes(self) -> int | None:
        """Compatibility alias for clients that used the old undifferentiated duration."""
        return self.preparation_minutes


class RecipeSource(BaseModel):
    name: str = Field(description = "Nom court de la source")
    description: str | None = Field(default = None, description = "Contexte optionnel de la source")
    uri: str | None = Field(default = None, description = "URL publique de la source")


class RecipeComponentPlan(BaseModel):
    name: str = Field(description = "Nom de la sous-recette (ex. Meringue, Caramel, Crème anglaise)")
    description: str | None = Field(default = None, description = "Description courte de la sous-recette")
    servings: str | None = Field(default = None, description = "Nombre de portions de la sous-recette")
    prep_time_minutes: int | None = Field(default = None, description = "Temps de préparation en minutes")
    cook_time_minutes: int | None = Field(default = None, description = "Temps de cuisson en minutes")
    ingredients: list[IngredientLine] = Field(default_factory = list)
    ustensils: list[UstensilLine] = Field(default_factory = list)
    steps: list[RecipeStep] = Field(default_factory = list)
    servings_multiplier: float = Field(default = 1.0, gt = 0)


class RecipePlan(BaseModel):
    name: str = Field(description = "Nom de la recette")
    description: str | None = Field(default = None, description = "Description courte de la recette")
    servings: str | None = Field(default = None, description = "Nombre de portions (ex. 60 biscuits, 4 personnes)")
    prep_time_minutes: int | None = Field(default = None, description = "Temps de préparation en minutes")
    cook_time_minutes: int | None = Field(default = None, description = "Temps de cuisson en minutes")
    origin_country: str | None = Field(
        default = None,
        pattern = r"^[A-Z]{2}$",
        description = "Pays d'origine probable au format ISO 3166-1 alpha-2",
    )
    difficulty: int | None = Field(default = None, ge = 1, le = 5, description = "Difficulté de 1 à 5")
    price: int | None = Field(default = None, ge = 1, le = 5, description = "Niveau de prix de 1 à 5")
    sources: list[RecipeSource] = Field(default_factory = list)
    tags: list[str] = Field(
        default_factory=list,
        description="Noms de tags pertinents choisis exclusivement dans le catalogue fourni",
    )
    ingredients: list[IngredientLine] = Field(default_factory = list)
    ustensils: list[UstensilLine] = Field(default_factory = list)
    steps: list[RecipeStep] = Field(default_factory = list)
    components: list[RecipeComponentPlan] = Field(default_factory = list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_time_minutes(self) -> int:
        step_total = sum(step.total_minutes for step in self.steps)
        return step_total or (self.prep_time_minutes or 0) + (self.cook_time_minutes or 0)


class RecipeResult(BaseModel):
    recipe_uuid: str
    recipe_name: str
    resolved_ingredients: dict[str, str]  # name → uuid
    resolved_ustensils: dict[str, str]  # name → uuid
    formatted_response: str
    created: bool = True
    duplicate_confirmation_required: bool = False
    existing_recipe_uuid: str | None = None
    existing_recipe_name: str | None = None
    updated: bool = False
    image_generated: bool = False
    warnings: list[str] = Field(default_factory = list)
