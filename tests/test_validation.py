from __future__ import annotations

from synthetic_data_engine.tasks.validation import validate_item


def test_validate_item_accepts_matching_schema():
    schema = {
        "type": "object",
        "required": ["prompt", "answer", "metadata"],
        "properties": {
            "prompt": {"type": "string"},
            "answer": {"type": "string"},
            "metadata": {
                "type": "object",
                "required": ["difficulty"],
                "properties": {
                    "difficulty": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }

    result = validate_item(
        schema,
        {
            "prompt": "Explain this.",
            "answer": "A concrete answer.",
            "metadata": {"difficulty": "easy", "tags": ["reasoning"]},
        },
    )

    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_item_rejects_missing_and_wrong_types():
    schema = {
        "type": "object",
        "required": ["prompt", "answer"],
        "properties": {
            "prompt": {"type": "string"},
            "answer": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }

    result = validate_item(schema, {"prompt": 123, "metadata": []})

    assert result["valid"] is False
    assert "$.answer is required" in result["errors"]
    assert "$.prompt expected string, got integer" in result["errors"]
    assert "$.metadata expected object, got array" in result["errors"]
