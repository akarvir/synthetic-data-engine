from __future__ import annotations

import pytest

from synthetic_data_engine.models.factory import create_model
from synthetic_data_engine.models.ollama import OllamaModel


def test_create_ollama_model_uses_local_defaults():
    model = create_model("ollama", "llama3.1")

    assert isinstance(model, OllamaModel)
    assert model.name == "ollama:llama3.1"
    assert model.model == "llama3.1"
    assert model.api_key == "ollama"
    assert model.base_url == "http://localhost:11434/v1"


def test_ollama_provider_requires_model_name():
    with pytest.raises(ValueError, match="--model is required for ollama"):
        create_model("ollama")
