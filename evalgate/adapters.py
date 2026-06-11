"""Adapters for calling LLM endpoints (OpenAI-compatible API)."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512


class CompletionResponse(BaseModel):
    id: str = ""
    model: str = ""
    content: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class OpenAIAdapter:
    """Thin adapter for OpenAI-compatible /v1/chat/completions endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a chat completion request and return the parsed response."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content,
            usage=data.get("usage", {}),
        )
