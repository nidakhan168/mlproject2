from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from numbers import Number
from typing import Literal

from .result import ArrayResult, SelectionError

__all__ = [
    "get_array",
    "select_array",
    "analyze_array",
    "stream_array",
]


def _try_array(type_code: str, values: Iterable) -> array | None:
    """Attempt to build an array with the given type code.
    Return the array if successful, otherwise None.
    """
    try:
        return array(type_code, values)
    except (OverflowError, TypeError, ValueError):
        return None


def _is_all_numeric(values: Iterable) -> bool:
    return all(isinstance(x, Number) for x in values)


def _min_max(values: list[float]) -> tuple[float, float]:
    # Native min/max to avoid external dependency
    return (min(values), max(values))  # pragma: no cover - legacy helper


def get_array(iterable: Iterable) -> array | list:
    """Backward-compatible API delegating to the policy engine.

    Equivalent to ``select_array(iterable, policy='smallest')``.
    """
    return select_array(iterable, policy="smallest")


def _scan_stats(values: list[float]) -> tuple[int, float | None, float | None, bool]:
    count = len(values)
    if count == 0:  # pragma: no cover - callers guard empty before calling
        return 0, None, None, False
    has_float = any(isinstance(v, float) for v in values)
    try:
        vmin = min(values)
        vmax = max(values)
    except ValueError:  # pragma: no cover - guarded by count check
        vmin, vmax = None, None
    return count, vmin, vmax, has_float


def _type_sequence(
    policy: Literal["smallest", "balanced", "wide"],
    prefer_signed: bool,
    no_float: bool,
    min_type: str | None,
    max_type: str | None,
) -> list[str]:
    int_order_signed_first = ["b", "B", "h", "H", "i", "I", "q", "Q"]
    int_order_unsigned_first = ["B", "b", "H", "h", "I", "i", "Q", "q"]
    base = int_order_signed_first if prefer_signed else int_order_unsigned_first

    if policy == "balanced":
        base = ["b", "B", "h", "H", "i", "I", "q", "Q"]
    elif policy == "wide":
        base = ["i", "I", "q", "Q", "h", "H", "b", "B"]

    seq = list(base)
    if not no_float:
        seq.extend(["f", "d"])  # floats last

    # Bound by min_type/max_type if provided
    if min_type or max_type:
        order = {
            code: idx for idx, code in enumerate(["b", "B", "h", "H", "i", "I", "q", "Q", "f", "d"])
        }
        lo = order[min_type] if (min_type is not None and min_type in order) else 0
        hi = (
            order[max_type] if (max_type is not None and max_type in order) else max(order.values())
        )
        seq = [c for c in seq if lo <= order.get(c, lo) <= hi]

    return seq


def select_array(
    iterable: Iterable,
    *,
    policy: Literal["smallest", "balanced", "wide"] = "smallest",
    min_type: str | None = None,
    max_type: str | None = None,
    prefer_signed: bool = False,
    allow_float_downgrade: bool = True,
    no_float: bool = False,
    strict: bool = False,
) -> array | list:
    values = list(iterable)
    if len(values) == 0:
        return []
    if not _is_all_numeric(values):
        return list(values)

    floats = [float(v) for v in values]
    count, vmin, vmax, has_float = _scan_stats(floats)

    # Build candidate order
    candidates = _type_sequence(policy, prefer_signed, no_float, min_type, max_type)

    # If floats present and no_float, must return list or raise
    if has_float and no_float:
        if strict:
            offending = next(v for v in values if isinstance(v, float))
            raise SelectionError(0, offending, "integers only")
        return list(values)

    # For float32 downgrade control
    F32_MAX = 3.4028235e38
    if (
        not allow_float_downgrade
        and has_float
        and (vmax is not None)
        and (abs(vmax) > F32_MAX or (vmin is not None and abs(vmin) > F32_MAX))
    ):
        # Skip 'f' if values exceed float32 range
        candidates = [c for c in candidates if c != "f"]

    # Try candidates in order
    for code in candidates:
        if code in {"f", "d"} and has_float:
            if code == "f":
                # Skip if out of finite float32 range
                if vmax is not None and (
                    abs(vmax) > F32_MAX or (vmin is not None and abs(vmin) > F32_MAX)
                ):
                    continue  # pragma: no cover - covered by tests but may elide in branch coverage
        a = _try_array(code, values)
        if a is not None:
            return a

    if strict:
        # Find the first violating index with a helpful message
        try_code = candidates[-1] if candidates else "d"
        for idx, val in enumerate(values):
            try:
                array(try_code, [val])
            except Exception:
                raise SelectionError(idx, val, try_code)
        raise SelectionError(0, values[0], try_code)

    return list(values)


def analyze_array(
    iterable: Iterable,
    **kwargs,
) -> ArrayResult:
    values = list(iterable)
    if len(values) == 0:
        return ArrayResult(value=[], typecode=None, count=0, min=None, max=None, reason="empty")
    if not _is_all_numeric(values):
        return ArrayResult(
            value=list(values),
            typecode=None,
            count=len(values),
            min=None,
            max=None,
            reason="non-numeric",
        )
    floats = [float(v) for v in values]
    count, vmin, vmax, has_float = _scan_stats(floats)
    out = select_array(values, **kwargs)
    if isinstance(out, array):
        return ArrayResult(
            value=out,
            typecode=out.typecode,
            count=count,
            min=vmin,
            max=vmax,
            reason=("float" if has_float else "int"),
        )
    return ArrayResult(
        value=list(values),
        typecode=None,
        count=count,
        min=vmin,
        max=vmax,
        reason="no-fitting-type",
    )


def stream_array(
    iterable: Iterable,
    *,
    chunk_size: int = 65536,
    **kwargs,
) -> Iterator[array | list]:
    """
    Yield arrays (or a list) chunk-by-chunk using the selection policy decided
    from the first chunk.

    If later chunks violate the selected type, yield a list for those chunks.
    """
    it = iter(iterable)
    first_chunk: list = []
    for _ in range(chunk_size):
        try:
            first_chunk.append(next(it))
        except StopIteration:
            break
    if not first_chunk:
        return

    selected = select_array(first_chunk, **kwargs)
    yield selected

    # Determine typecode if array
    code: str | None = selected.typecode if isinstance(selected, array) else None

    buf: list = []
    for item in it:
        buf.append(item)
        if len(buf) >= chunk_size:
            if code is None:
                yield list(buf)
            else:
                a = _try_array(code, buf)
                yield a if a is not None else list(buf)
            buf.clear()
    if buf:
        if code is None:
            yield list(buf)
        else:
            a = _try_array(code, buf)
            yield a if a is not None else list(buf)
