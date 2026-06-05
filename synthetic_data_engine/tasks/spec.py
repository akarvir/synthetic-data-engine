from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    description: str
    output_schema: dict[str, Any]
    requirements: list[str] = field(default_factory=list)
    difficulty: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    topics: list[str] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TaskSpec":
        missing = [key for key in ("name", "domain", "description", "output_schema") if key not in data]
        if missing:
            raise ValueError(f"Task spec is missing required field(s): {', '.join(missing)}")

        schema = data["output_schema"]
        if not isinstance(schema, dict):
            raise ValueError("Task spec output_schema must be an object")

        return cls(
            name=str(data["name"]),
            domain=str(data["domain"]),
            description=str(data["description"]),
            output_schema=schema,
            requirements=[str(item) for item in data.get("requirements", [])],
            difficulty=[str(item) for item in data.get("difficulty", ["easy", "medium", "hard"])],
            topics=[str(item) for item in data.get("topics", [])],
            selection=dict(data.get("selection", {})),
        )

    @property
    def required_fields(self) -> list[str]:
        return [str(field) for field in self.output_schema.get("required", [])]

    @property
    def min_score(self) -> float:
        return float(self.selection.get("min_score", 0.8))

    @property
    def max_items(self) -> int | None:
        value = self.selection.get("max_items")
        if value is None:
            return None
        return int(value)

    @property
    def difficulty_distribution(self) -> dict[str, float]:
        value = self.selection.get("difficulty_distribution", {})
        if not isinstance(value, dict):
            return {}
        return {str(key): float(weight) for key, weight in value.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "output_schema": self.output_schema,
            "requirements": self.requirements,
            "difficulty": self.difficulty,
            "topics": self.topics,
            "selection": self.selection,
        }
