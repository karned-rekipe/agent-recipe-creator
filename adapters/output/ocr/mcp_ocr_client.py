import json
from typing import Any

import fastmcp
from fastmcp.client.transports import StreamableHttpTransport

from domain.ports.recipe_ocr import RecipeOCRPort


class MCPOCRClient(RecipeOCRPort):
    def __init__(self, url: str, timeout_seconds: float = 120.0) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def extract(
        self,
        *,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_media_type: str | None = None,
    ) -> str:
        transport = StreamableHttpTransport(self._url)
        async with fastmcp.Client(transport, timeout=self._timeout_seconds) as client:
            result = await client.call_tool(
                "extract_recipe_text",
                {
                    "image_url": image_url,
                    "image_base64": image_base64,
                    "image_media_type": image_media_type,
                },
                timeout=self._timeout_seconds,
            )
        payload = _result_data(result)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise RuntimeError("OCR service returned an invalid response")
        text = payload["text"].strip()
        if len(text) < 10:
            raise RuntimeError("OCR service did not extract usable recipe text")
        return text


def _result_data(result: Any) -> Any:
    if result.data is not None:
        return result.data
    if result.structured_content:
        if set(result.structured_content) == {"result"}:
            return result.structured_content["result"]
        return result.structured_content
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    return None
