from __future__ import annotations

from array import array
from dataclasses import dataclass


@dataclass
class ArrayResult:
    value: array | list
    typecode: str | None
    count: int
    min: float | None
    max: float | None
    reason: str


class SelectionError(ValueError):
    def __init__(self, index: int, value, expected: str):
        super().__init__(f"Value at index {index}={value!r} violates expected {expected}")
        self.index = index
        self.value = value
        self.expected = expected
