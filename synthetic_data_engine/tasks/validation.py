from __future__ import annotations

from typing import Any


def validate_item(schema: dict[str, Any], item: Any) -> dict[str, Any]:
    errors = _validate_value(schema, item, path="$")
    if errors:
        return {
            "valid": False,
            "reason": "; ".join(errors),
            "errors": errors,
        }
    return {
        "valid": True,
        "reason": "Candidate matches the task schema.",
        "errors": [],
    }


def _validate_value(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    expected_type = schema.get("type")
    if expected_type and not _matches_type(expected_type, value):
        return [f"{path} expected {expected_type}, got {_type_name(value)}"]

    if expected_type == "object" or "properties" in schema:
        if not isinstance(value, dict):
            return [f"{path} expected object, got {_type_name(value)}"]
        return _validate_object(schema, value, path)

    if expected_type == "array":
        if not isinstance(value, list):
            return [f"{path} expected array, got {_type_name(value)}"]
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return []
        errors: list[str] = []
        for index, item in enumerate(value):
            errors.extend(_validate_value(item_schema, item, f"{path}[{index}]"))
        return errors

    return []


def _validate_object(schema: dict[str, Any], value: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    for field in schema.get("required", []):
        if field not in value or value[field] in ("", None, []):
            errors.append(f"{path}.{field} is required")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return errors

    for field, field_schema in properties.items():
        if field not in value:
            continue
        if isinstance(field_schema, dict):
            errors.extend(_validate_value(field_schema, value[field], f"{path}.{field}"))

    return errors


def _matches_type(expected_type: str | list[str], value: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(item, value) for item in expected_type)

    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__
