from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def item_hash(item: dict[str, Any]) -> str:
    normalized = normalize_item(item)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_item(item: dict[str, Any]) -> str:
    serialized = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", serialized).strip().lower()
