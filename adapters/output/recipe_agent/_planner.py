from collections.abc import Mapping
from typing import cast

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelProfile, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from adapters.output.recipe_agent._logger import log as logger
from domain.errors import AIProviderCreditExhaustedError
from domain.models.recipe import RecipePlan
from infrastructure.config import LMSettings

_SYSTEM_PROMPT = (
    "Tu es un assistant d'extraction culinaire. "
    "À partir d'une description en langage naturel, tu structures une recette complète. "
    "\n\n"
    "## Nom de la recette\n"
    "Détermine un nom court, canonique et normalisé (ex. 'Tarte Tatin', 'Pizza Margherita', 'Soupe à l'oignon'). "
    "Ne préfixe jamais avec 'Recette de' ou 'La recette'. "
    "Ce nom sera utilisé pour rechercher si la recette existe déjà — sois précis et cohérent.\n\n"
    "## Ingrédients\n"
    "Identifie tous les ingrédients avec leur quantité et leur unité si mentionnées. "
    "Si la quantité ou l'unité est implicite, propose une valeur utilisable (ex. '1' + 'pincée', "
    "'1' + 'pièce', '1' + 'unité') plutôt que de laisser vide. "
    "Normalise les noms (ex. 'farine de blé', 'sel fin') sans majuscule inutile.\n\n"
    "## Ustensiles\n"
    "Identifie tous les ustensiles nécessaires à la réalisation. "
    "Normalise les noms (ex. 'fouet', 'casserole 20 cm').\n\n"
    "## Étapes\n"
    "Décompose la recette en étapes ordonnées : donne un titre court et une instruction complète à chaque étape, "
    "ainsi que les durées de préparation active, cuisson et repos en minutes si elles sont précisées ou déductibles. "
    "Ventile le temps total annoncé entre les étapes et les bonnes catégories sans le compter deux fois. "
    "Dans chaque étape, utilise uniquement des ingrédients et ustensiles présents dans les listes.\n\n"
    "## Recettes composées\n"
    "Quand la recette est structurée en préparations distinctes (ex. meringue, caramel, crème anglaise), "
    "renseigne chaque préparation dans `components` avec ses propres ingrédients, ustensiles et étapes. "
    "La recette principale doit alors contenir seulement les étapes d'assemblage globales si elles existent.\n\n"
    "## Métadonnées\n"
    "Extrais le nombre de portions et les temps de préparation/cuisson. "
    "Si l'origine, la difficulté ou le prix ne sont pas explicitement écrits, estime-les à partir du plat, "
    "des techniques, des ingrédients et du contexte. Utilise un code pays ISO alpha-2 et des scores de 1 à 5. "
    "Conserve les sources identifiables. La saison sera calculée par le registre depuis les ingrédients.\n\n"
    "## Tags\n"
    "Choisis uniquement les tags réellement pertinents dans le catalogue fourni avec la demande. "
    "Recopie leur nom exactement, sans inventer ni reformuler de tag. "
    "Un tag seulement vaguement lié à la recette doit être omis.\n\n"
    "## Description\n"
    "La description finale doit être un court résumé éditorial. N'y laisse aucune liste d'ingrédients ni étape : "
    "reclasse systématiquement ces éléments dans ingredients et steps.\n\n"
    "Réponds toujours en français."
)

_REVIEW_PROMPT = (
    "Tu es un relecteur culinaire qualité. "
    "Tu reçois le texte brut et un JSON de recette déjà extrait. "
    "Retourne une version corrigée du même schéma RecipePlan.\n\n"
    "Objectif principal : cohérence ingrédients ↔ étapes.\n"
    "- Chaque ingrédient explicitement utilisé dans une étape doit exister dans la liste d'ingrédients de la recette ou de sa sous-recette.\n"
    "- Chaque étape doit utiliser les noms canoniques présents dans la liste d'ingrédients.\n"
    "- Si un ingrédient manque mais que sa quantité est inconnue, ajoute-le avec quantity='1' et unit='unité'.\n"
    "- Retire de description les ingrédients et instructions qui y étaient mal rangés après les avoir reclassés.\n"
    "- Complète origin_country, difficulty et price lorsqu'ils sont absents, sans remplacer une valeur existante fournie.\n"
    "- Vérifie la ventilation des temps entre préparation, cuisson et repos pour permettre un total fiable.\n"
    "- Conserve uniquement les tags pertinents qui appartiennent exactement au catalogue fourni.\n"
    "- Ne supprime pas une sous-recette utile.\n"
    "- Garde une réponse en français et ne renvoie que le JSON structuré."
)

# 'prompted' avoids tool-calling (unreliable on local LLMs like Qwen3/LLaMA).
# openai_chat_send_back_thinking_parts=False prevents thinking tokens from
# accumulating in retry context and blowing the context window.
_LOCAL_PROFILE = OpenAIModelProfile(
    default_structured_output_mode = "prompted",
    openai_chat_send_back_thinking_parts = False,
)

_OPENAI_CREDIT_EXHAUSTION_CODES = frozenset({"credit_balance_exhausted", "insufficient_quota"})


def _is_openai_credit_exhausted(exc: ModelHTTPError) -> bool:
    if exc.status_code != 429 or not isinstance(exc.body, Mapping):
        return False

    body = exc.body.get("error", exc.body)
    if not isinstance(body, Mapping):
        return False

    provider_codes = {
        value
        for key in ("code", "type")
        if isinstance((value := body.get(key)), str)
    }
    return bool(provider_codes & _OPENAI_CREDIT_EXHAUSTION_CODES)


async def _run_with_provider_error_mapping(agent: Agent[None, RecipePlan], prompt: str):
    try:
        return await agent.run(prompt)
    except ModelHTTPError as exc:
        if _is_openai_credit_exhausted(exc):
            raise AIProviderCreditExhaustedError() from exc
        raise


def _build_model(settings: LMSettings):
    if settings.provider == "anthropic":
        logger.debug(f"planner provider=anthropic model={settings.model_name}")
        return AnthropicModel(
            settings.model_name,
            provider = AnthropicProvider(api_key = settings.api_key),
        )
    provider = OpenAIProvider(base_url = settings.base_url, api_key = settings.api_key)
    if settings.base_url:
        logger.debug(f"planner provider=openai-local model={settings.model_name} base_url={settings.base_url}")
        return OpenAIChatModel(
            settings.model_name,
            provider = provider,
            profile = _LOCAL_PROFILE,
        )

    logger.debug(f"planner provider=openai model={settings.model_name}")
    return OpenAIResponsesModel(settings.model_name, provider = provider)


class _PydanticAIPlanner:
    def __init__(self, settings: LMSettings) -> None:
        self._agent = cast(Agent[None, RecipePlan], Agent(
            _build_model(settings),
            output_type=RecipePlan,
            system_prompt=_SYSTEM_PROMPT,
            retries=3,
        ))
        self._review_agent = cast(Agent[None, RecipePlan], Agent(
            _build_model(settings),
            output_type=RecipePlan,
            system_prompt=_REVIEW_PROMPT,
            retries=3,
        ))

    async def plan(self, user_input: str, available_tags: list[str] | None = None) -> RecipePlan:
        result = await _run_with_provider_error_mapping(
            self._agent,
            _with_tag_catalog(user_input, available_tags),
        )
        self._log_messages(result)
        usage = result.usage()
        logger.debug(
            f"  LLM usage requests={usage.requests} "
            f"input={usage.input_tokens} output={usage.output_tokens} total={usage.total_tokens}"
        )
        return result.output

    async def review(
        self,
        raw_text: str,
        plan: RecipePlan,
        available_tags: list[str] | None = None,
    ) -> RecipePlan:
        review_input = (
            "TEXTE BRUT:\n"
            f"{raw_text}\n\n"
            "RECETTE EXTRAITE JSON:\n"
            f"{plan.model_dump_json()}\n\n"
            f"CATALOGUE DE TAGS AUTORISÉS:\n{_format_tag_catalog(available_tags)}"
        )
        result = await _run_with_provider_error_mapping(self._review_agent, review_input)
        self._log_messages(result)
        usage = result.usage()
        logger.debug(
            f"  review LLM usage requests={usage.requests} "
            f"input={usage.input_tokens} output={usage.output_tokens} total={usage.total_tokens}"
        )
        return result.output

    @staticmethod
    def _log_messages(result) -> None:
        for msg in result.all_messages():
            for part in getattr(msg, "parts", []):
                kind = type(part).__name__
                match kind:
                    case "ThinkingPart":
                        logger.debug(f"  LLM thinking ({len(part.content)} chars): {part.content[:300]!r}")
                    case "TextPart":
                        logger.debug(f"  LLM text: {part.content[:500]!r}")
                    case "ToolCallPart":
                        logger.debug(f"  LLM tool_call name={part.tool_name} args={str(part.args)[:300]}")


def _format_tag_catalog(available_tags: list[str] | None) -> str:
    if not available_tags:
        return "(aucun tag disponible : retourne une liste vide)"
    return "\n".join(f"- {name}" for name in available_tags)


def _with_tag_catalog(user_input: str, available_tags: list[str] | None) -> str:
    return (
        f"{user_input}\n\n"
        "CATALOGUE DE TAGS AUTORISÉS (choisir uniquement dans cette liste) :\n"
        f"{_format_tag_catalog(available_tags)}"
    )
