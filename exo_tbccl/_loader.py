"""Loads the native binding and wires the structured-error factory. Import failures stay ImportError (exo treats them as 'backend unavailable')."""

from . import _native  # pyright: ignore[reportAttributeAccessIssue]
from .errors import error_for

_native.set_error_factory(error_for)

native = _native
