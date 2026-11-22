import re
import unicodedata

from adapters.output.recipe_agent._logger import log as logger
from rapidfuzz import fuzz, process


def _normalize_name(value: str) -> str:
    value = value.casefold().replace("œ", "oe").replace("æ", "ae")
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"\w+", value))


def _lookup_variants(name: str) -> list[str]:
    normalized = _normalize_name(name)
    variants = [name]
    if normalized and normalized != name.casefold():
        variants.append(normalized)
    if " " not in normalized and len(normalized) > 3 and normalized.endswith("s"):
        variants.append(normalized[:-1])
    return list(dict.fromkeys(variants))


def _make_fuzzy_matcher(threshold: int):
    def fuzzy_match(name: str, candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        normalized_name = _normalize_name(name)
        names = [_normalize_name(c["name"]) for c in candidates]
        result = process.extractOne(normalized_name, names, scorer = fuzz.ratio)
        if result and result[1] >= threshold:
            matched = candidates[result[2]]
            logger.debug(
                f"  fuzzy name={name!r} → MATCH {matched['name']!r} "
                f"score={result[1]} uuid={matched['uuid']}"
            )
            return matched
        best_name = candidates[result[2]]["name"] if result else "—"
        best_score = result[1] if result else 0
        logger.debug(f"  fuzzy name={name!r} → no match (best={best_name!r} score={best_score})")
        return None

    return fuzzy_match
