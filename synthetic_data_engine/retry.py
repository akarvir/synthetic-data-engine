from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


async def with_retries(operation: Callable[[], Awaitable[T]], retries: int, delay_seconds: float = 0.25) -> T:
    attempts = max(1, retries + 1)
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(delay_seconds * (2**attempt))
    raise last_error if last_error is not None else RuntimeError("Retry operation failed without an exception.")
