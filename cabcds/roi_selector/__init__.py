"""ROI selector module.

Exports the main config and pipeline entrypoints for convenient imports.
"""

from .config import ROISelectorConfig, RoiSelectorFeatureConfig, load_roi_selector_config
from .inference import RoiSelector
from .training import RoiSelectorTrainer

__all__ = [
    "ROISelectorConfig",
    "RoiSelectorFeatureConfig",
    "load_roi_selector_config",
    "RoiSelector",
    "RoiSelectorTrainer",
]
