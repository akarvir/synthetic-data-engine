from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


Message = dict[str, str]


class ModelClient(ABC):
    name: str

    @abstractmethod
    async def complete_json(self, messages: list[Message], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a JSON object generated from chat-style messages."""
