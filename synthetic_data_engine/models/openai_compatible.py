from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from synthetic_data_engine.models.base import Message, ModelClient


class OpenAICompatibleModel(ModelClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.name = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai-compatible provider")

    async def complete_json(self, messages: list[Message], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._complete_json_sync, messages, schema)

    def _complete_json_sync(self, messages: list[Message], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.8,
            "response_format": {"type": "json_object"},
        }
        if schema:
            body["metadata"] = {"schema_name": schema.get("title", "synthetic_data_item")}

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed with HTTP {exc.code}: {detail}") from exc

        content = response_body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Model returned JSON, but not a JSON object")
        return parsed
