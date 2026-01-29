"""Core package for the CABCDS reproduction project.

Keep imports lightweight to avoid pulling in heavy dependencies (e.g., torch)
for workflows that only need data preparation or classical ML utilities.
"""

from __future__ import annotations

__all__ = ["RoiSelectorTrainer"]


def __getattr__(name: str):
	if name == "RoiSelectorTrainer":
		from cabcds.roi_selector.training import RoiSelectorTrainer  # local import (lazy)

		return RoiSelectorTrainer
	raise AttributeError(name)
