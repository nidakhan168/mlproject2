from toarray.iterable_to_array import (
    analyze_array as analyze_array,
)
from toarray.iterable_to_array import (
    get_array as get_array,
)
from toarray.iterable_to_array import (
    select_array as select_array,
)
from toarray.iterable_to_array import (
    stream_array as stream_array,
)

# Optional interop modules
try:
    from .numpy_compat import to_numpy as to_numpy
except Exception:  # pragma: no cover
    to_numpy = None  # type: ignore[assignment]

try:
    from .arrow_compat import to_arrow as to_arrow
except Exception:  # pragma: no cover
    to_arrow = None  # type: ignore[assignment]

__all__ = [
    "get_array",
    "select_array",
    "analyze_array",
    "stream_array",
    "to_numpy",
    "to_arrow",
    "__version__",
]

__version__ = "0.3.2"
