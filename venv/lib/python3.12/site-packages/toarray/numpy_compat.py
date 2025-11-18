from __future__ import annotations

from array import array
from collections.abc import Iterable
from typing import Any, Literal

from .iterable_to_array import select_array

np: Any
try:
    import numpy as _np

    np = _np
except Exception:  # pragma: no cover - import-guarded
    np = None


def to_numpy(
    iterable: Iterable,
    dtype: Literal["min", "float32", "float64", "int64", "uint64"] = "min",
    copy: bool = False,
    **policy,
):
    if np is None:  # pragma: no cover - import-guarded path
        raise RuntimeError("NumPy not available. Install with `pip install toarray[numpy]`.")

    if dtype == "min":
        out = select_array(iterable, **policy)
        if isinstance(out, array):
            # Zero-copy for numeric arrays
            return np.frombuffer(out, dtype=_typecode_to_numpy_dtype(out.typecode))
        # Fallback to object array
        return np.array(list(out), dtype=object)

    # Explicit dtype path
    if dtype in {"float32", "float64", "int64", "uint64"}:
        np_dtype = {
            "float32": np.float32,
            "float64": np.float64,
            "int64": np.int64,
            "uint64": np.uint64,
        }[dtype]
        return np.asarray(list(iterable), dtype=np_dtype)

    raise ValueError(f"Unsupported dtype: {dtype}")


def _typecode_to_numpy_dtype(code: str):
    import numpy as _np  # local import

    return {
        "b": _np.int8,
        "B": _np.uint8,
        "h": _np.int16,
        "H": _np.uint16,
        "i": _np.int32,
        "I": _np.uint32,
        "q": _np.int64,
        "Q": _np.uint64,
        "f": _np.float32,
        "d": _np.float64,
    }[code]
