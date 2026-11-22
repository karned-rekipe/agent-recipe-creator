from pydantic import BaseModel, Field, model_validator

from domain.models.recipe_intake import RecipeIntake


class AiCreateRequestSchema(BaseModel):
    raw_text: str | None = Field(
        None,
        description = "Texte brut de la recette : copier/coller internet, OCR, saisie libre…",
        examples = ["Tarte tatin : 6 pommes, 150g beurre, 200g sucre…"],
        min_length = 10,
    )
    source_url: str | None = Field(None, description="URL publique d'une fiche recette Web")
    image_url: str | None = Field(None, description="URL publique d'une image contenant une recette")
    image_base64: str | None = Field(None, description="Image encodée en base64, sans préfixe data URL")
    image_media_type: str | None = Field(None, description="Type MIME de image_base64, par exemple image/jpeg")
    allow_duplicate: bool = Field(
        False,
        description = "Créer un doublon si une recette équivalente existe déjà.",
    )
    generate_image: bool = Field(True, description="Générer automatiquement l'image principale après création")

    @model_validator(mode="after")
    def validate_source(self) -> "AiCreateRequestSchema":
        self.to_domain()
        return self

    def to_domain(self) -> RecipeIntake:
        return RecipeIntake(
            raw_text=self.raw_text,
            source_url=self.source_url,
            image_url=self.image_url,
            image_base64=self.image_base64,
            image_media_type=self.image_media_type,
        )


class AiEnrichRequestSchema(BaseModel):
    generate_image: bool = Field(True, description="Générer l'image principale lorsqu'elle est absente")


class AiCreateResponseSchema(BaseModel):
    recipe_uuid: str = Field(description = "UUID de la recette créée ou de la recette existante à confirmer")
    recipe_name: str = Field(description = "Nom de la recette structurée par l'agent")
    formatted_response: str = Field(description = "Résumé Markdown de la recette créée")
    created: bool = Field(default = True, description = "Indique si une nouvelle recette a été créée")
    duplicate_confirmation_required: bool = Field(
        default = False,
        description = "Indique si l'utilisateur doit confirmer la création d'un doublon",
    )
    existing_recipe_uuid: str | None = Field(default = None, description = "UUID de la recette déjà existante")
    existing_recipe_name: str | None = Field(default = None, description = "Nom de la recette déjà existante")
    updated: bool = Field(default=False, description="Indique qu'une fiche existante a été mise à niveau")
    image_generated: bool = Field(default=False, description="Indique qu'une image principale a été générée")
    warnings: list[str] = Field(default_factory=list, description="Avertissements non bloquants")
