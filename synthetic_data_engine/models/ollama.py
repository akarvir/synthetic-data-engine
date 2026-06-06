from __future__ import annotations

import os

from synthetic_data_engine.models.openai_compatible import OpenAICompatibleModel


class OllamaModel(OpenAICompatibleModel):
    def __init__(self, model: str, base_url: str | None = None, timeout: int = 120) -> None:
        super().__init__(
            model=model,
            api_key="ollama",
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
            timeout=timeout,
        )
        self.name = f"ollama:{model}"
