"""Helpers shared by the provider tests."""
import sys
import unittest
from pathlib import Path

PROVIDER_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(PROVIDER_DIR) not in sys.path:
    sys.path.insert(0, str(PROVIDER_DIR))


class approx:
    """A small stand-in for pytest.approx so the tests need only the standard library: a float, or a sequence of floats, within a relative and an absolute tolerance."""

    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def _close(self, a, b):
        return a == b or abs(a - b) <= max(self.rel * max(abs(a), abs(b)), self.abs)

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            return isinstance(other, (list, tuple)) and len(other) == len(self.expected) and all(self._close(a, b) for a, b in zip(other, self.expected))
        return isinstance(other, (int, float)) and self._close(other, self.expected)

    def __repr__(self):
        return f"approx({self.expected!r})"
