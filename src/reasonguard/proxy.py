"""LLM proxy layer — forward requests to upstream OpenAI-compatible API."""

import httpx

from .config import settings


async def forward_chat_completion(request_body: dict) -> dict:
    """Forward a chat completion request to the upstream LLM and return the JSON response."""
    url = f"{settings.upstream_base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=request_body, headers=headers)
        resp.raise_for_status()
        return resp.json()


def extract_content_from_response(response_data: dict) -> str:
    """Extract the assistant message content from an OpenAI-format response."""
    choices = response_data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "")
