from __future__ import annotations

from synthetic_data_engine.models.base import ModelClient
from synthetic_data_engine.models.local import LocalDeterministicModel
from synthetic_data_engine.models.openai_compatible import OpenAICompatibleModel


def create_model(provider: str, model: str | None = None) -> ModelClient:
    if provider == "local":
        return LocalDeterministicModel()
    if provider == "openai-compatible":
        if not model:
            raise ValueError("--model is required for openai-compatible")
        return OpenAICompatibleModel(model=model)
    raise ValueError(f"Unsupported model provider: {provider}")
