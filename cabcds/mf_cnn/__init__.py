"""MF-CNN model components.

This package currently provides Torchvision-based reference implementations of:
- `CNNSeg`: VGG16-based FCN-style segmentation network
- `CNNDet`: AlexNet-based mitosis detector
- `CNNGlobal`: AlexNet-based ROI scoring classifier
"""

from cabcds.mf_cnn.config import MfcCnnConfig, load_mfc_cnn_config
from cabcds.mf_cnn.cnn import CNNSeg, CNNSegLegacy, CNNDet, CNNGlobal, SegmentationOutput
from cabcds.mf_cnn.checkpoints import load_cnn_seg_from_checkpoint
from cabcds.mf_cnn.preprocess import (
    GlobalPatchRecord,
    extract_global_patches_from_wsi,
    write_global_patch_index_csv,
)
from cabcds.mf_cnn.utils.loader import (
    GlobalScoringPatchDataset,
    MitosisDetectionDataset,
    MitosisSegmentationDataset,
    build_default_mf_cnn_train_loaders,
    load_tupac_train_scores,
)

__all__ = [
    "MfcCnnConfig",
    "CNNSeg",
    "CNNSegLegacy",
    "CNNDet",
    "CNNGlobal",
    "SegmentationOutput",
    "load_cnn_seg_from_checkpoint",
    "GlobalPatchRecord",
    "extract_global_patches_from_wsi",
    "write_global_patch_index_csv",
    "MitosisSegmentationDataset",
    "MitosisDetectionDataset",
    "GlobalScoringPatchDataset",
    "build_default_mf_cnn_train_loaders",
    "load_tupac_train_scores",
    "load_mfc_cnn_config",
]
