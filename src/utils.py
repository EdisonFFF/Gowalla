"""Small utility helpers for the project."""
from __future__ import annotations

import os
from typing import Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chunked_iterable(it: Iterable, size: int):
    it = iter(it)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk
