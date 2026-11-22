import asyncio
import ipaddress
import json
import socket
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from domain.models.recipe import RecipeSource
from domain.models.recipe_intake import ResolvedRecipeInput
from domain.ports.recipe_source_loader import RemoteRecipeDocument, RecipeSourceLoaderPort

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_TEXT_CHARS = 80_000
_MAX_REDIRECTS = 3
_OCR_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"}


class UnsafeRecipeURL(ValueError):
    pass


class _RecipeHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.json_ld_blocks: list[str] = []
        self.visible_parts: list[str] = []
        self._capture_json_ld = False
        self._json_ld_parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"style", "noscript"}:
            self._ignored_depth += 1
        if tag == "script":
            if (attributes.get("type") or "").lower().split(";")[0].strip() == "application/ld+json":
                self._capture_json_ld = True
                self._json_ld_parts = []
            else:
                self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._capture_json_ld:
            self.json_ld_blocks.append("".join(self._json_ld_parts))
            self._capture_json_ld = False
            self._json_ld_parts = []
            return
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capture_json_ld:
            self._json_ld_parts.append(data)
        elif not self._ignored_depth:
            cleaned = " ".join(data.split())
            if cleaned:
                self.visible_parts.append(cleaned)


class WebRecipeSourceLoader(RecipeSourceLoaderPort):
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        host_resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._transport = transport
        self._host_resolver = host_resolver or _resolve_host_addresses

    async def load(self, url: str) -> ResolvedRecipeInput:
        current_url = url.strip()
        async with httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=False,
            timeout=httpx.Timeout(15.0, connect=5.0),
            headers={"User-Agent": "Rekipe recipe importer/1.0"},
        ) as client:
            for redirect_count in range(_MAX_REDIRECTS + 1):
                await self._validate_public_url(current_url)
                response = await client.get(current_url)
                if response.is_redirect:
                    if redirect_count == _MAX_REDIRECTS:
                        raise ValueError("Too many redirects while loading recipe URL")
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Recipe URL redirect has no location")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                break

        content_type = response.headers.get("content-type", "").lower()
        media_type = content_type.split(";", 1)[0].strip()
        if media_type in _OCR_MEDIA_TYPES:
            raise RemoteRecipeDocument(current_url, media_type)
        if media_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("Recipe URL must return an HTML document")
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ValueError("Recipe page is too large")

        parser = _RecipeHTMLParser()
        parser.feed(response.text)
        recipe = _find_recipe_json_ld(parser.json_ld_blocks)
        extracted = _format_recipe(recipe) if recipe else "\n".join(parser.visible_parts)[:_MAX_TEXT_CHARS]
        if len(extracted.strip()) < 10:
            raise ValueError("No usable recipe content found at URL")

        parsed = urlparse(current_url)
        return ResolvedRecipeInput(
            raw_text=extracted[:_MAX_TEXT_CHARS],
            audit_label=f"recipe-url:{current_url}",
            sources=[RecipeSource(name=parsed.hostname or "Web", description="Recette importée depuis le Web", uri=current_url)],
        )

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise UnsafeRecipeURL("Recipe URL must use http or https")
        if parsed.username or parsed.password:
            raise UnsafeRecipeURL("Recipe URL credentials are not allowed")
        addresses = await asyncio.to_thread(self._host_resolver, parsed.hostname)
        if not addresses:
            raise UnsafeRecipeURL("Recipe URL hostname could not be resolved")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise UnsafeRecipeURL("Recipe URL must resolve only to public addresses")


def _resolve_host_addresses(hostname: str) -> list[str]:
    return sorted({str(item[4][0]) for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)})


def _find_recipe_json_ld(blocks: list[str]) -> dict[str, Any] | None:
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        for item in _walk_json(payload):
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if any(str(value).lower() == "recipe" for value in types):
                return item
    return None


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _instruction_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_instruction_text(item))
        return result
    if isinstance(value, dict):
        text = value.get("text") or value.get("name")
        nested = _instruction_text(value.get("itemListElement", []))
        return ([str(text)] if text else []) + nested
    return []


def _format_recipe(recipe: dict[str, Any]) -> str:
    lines = [f"Nom: {recipe.get('name', '')}"]
    for label, key in (
        ("Description", "description"),
        ("Portions", "recipeYield"),
        ("Cuisine/origine", "recipeCuisine"),
        ("Catégorie", "recipeCategory"),
        ("Préparation", "prepTime"),
        ("Cuisson", "cookTime"),
        ("Temps total", "totalTime"),
    ):
        if recipe.get(key):
            lines.append(f"{label}: {recipe[key]}")
    ingredients = recipe.get("recipeIngredient") or []
    if ingredients:
        lines.append("Ingrédients:")
        lines.extend(f"- {item}" for item in ingredients)
    instructions = _instruction_text(recipe.get("recipeInstructions", []))
    if instructions:
        lines.append("Étapes:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(instructions, 1))
    return "\n".join(lines)
