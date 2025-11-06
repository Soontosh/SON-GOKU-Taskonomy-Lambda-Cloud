"""Python package exposing the implementation in `pascal-context_eval/`.

This helper module makes the hyphenated directory importable as
`pascal_context_eval` so that the rest of the codebase can rely on standard
Python imports while keeping the file system layout requested by the project
specification.
"""
from __future__ import annotations

from pathlib import Path
import pkgutil

# Start with whatever pkgutil discovers for namespace packages.
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

# Also look inside the hyphenated directory where the actual implementation
# lives. This allows standard imports like ``pascal_context_eval.datasets``
# even though the code is stored under ``pascal-context_eval/``.
_hyphenated = Path(__file__).resolve().parent.parent / "pascal-context_eval"
if _hyphenated.is_dir():
    hyphenated_path = str(_hyphenated)
    if hyphenated_path not in __path__:
        __path__.append(hyphenated_path)  # type: ignore[attr-defined]

__all__ = []
