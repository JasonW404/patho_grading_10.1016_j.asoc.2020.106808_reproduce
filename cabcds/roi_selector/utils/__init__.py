"""ROI Selector utilities.

Keep this package lightweight on import. Some utilities (e.g., the training DataLoader)
depend on heavy optional deps like torch; we expose those lazily.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ROISelectorDataLoader"]


def __getattr__(name: str) -> Any:
	if name == "ROISelectorDataLoader":
		from .loader import ROISelectorDataLoader

		return ROISelectorDataLoader
	raise AttributeError(name)
