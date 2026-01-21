"""Pipeline utilities to build hybrid descriptors from ROI patches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, cast

import numpy as np
import torch
from skimage import measure
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cabcds.data_preparation.blue_ratio import compute_blob_stats
from cabcds.data_preparation.io import load_rgb_image
from cabcds.hybrid_descriptor.config import (
    HybridDescriptorConfig,
    HybridDescriptorInferenceConfig,
)
from cabcds.hybrid_descriptor.descriptor import HybridDescriptorBuilder, RoiMetrics
from cabcds.mf_cnn.mitosis_detection_net import MitosisDetectionNet
from cabcds.mf_cnn.mitosis_segmentation_net import MitosisSegmentationNet
from cabcds.mf_cnn.roi_scoring_net import RoiScoringNet


@dataclass(frozen=True)
class RoiPatchEntry:
    """Metadata for a ROI patch file.

    Attributes:
        path: Path to the patch image.
        group: Group identifier (typically WSI name).
    """

    path: Path
    group: str


class RoiPatchDataset(Dataset[RoiPatchEntry]):
    """Dataset of ROI patch entries."""

    def __init__(self, root_dir: Path, extensions: tuple[str, ...]) -> None:
        self.entries = _collect_roi_patches(root_dir, extensions)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> RoiPatchEntry:
        return self.entries[index]


class HybridDescriptorPipeline:
    """Build hybrid descriptors from ROI patches and MF-CNN outputs."""

    def __init__(
        self,
        hybrid_config: HybridDescriptorConfig,
        infer_config: HybridDescriptorInferenceConfig,
        segmentation_model: nn.Module | None = None,
        detection_model: nn.Module | None = None,
        roi_scoring_model: nn.Module | None = None,
    ) -> None:
        self.hybrid_config = hybrid_config
        self.infer_config = infer_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(infer_config.device)

        self.segmentation_model = segmentation_model or _load_segmentation_model(infer_config, self.device)
        self.detection_model = detection_model or _load_detection_model(infer_config, self.device)
        self.roi_scoring_model = roi_scoring_model or _load_roi_scoring_model(infer_config, self.device)

        self.roi_transform = _build_imagenet_transform(infer_config.roi_resize)
        self.det_transform = _build_imagenet_transform(infer_config.detection_resize)

    def run(self) -> dict[str, np.ndarray]:
        """Build hybrid descriptors for ROI patches.

        Returns:
            Mapping of group name to descriptor vector.
        """

        roi_root = Path(self.infer_config.roi_patches_dir)
        dataset = RoiPatchDataset(roi_root, self.infer_config.image_extensions)
        if len(dataset) == 0:
            raise FileNotFoundError(f"No ROI patches found in {roi_root}.")

        grouped_metrics = self._collect_roi_metrics(dataset)
        builder = HybridDescriptorBuilder(self.hybrid_config)
        descriptors = {group: builder.build(metrics) for group, metrics in grouped_metrics.items()}

        self._write_report(descriptors)
        return descriptors

    def _collect_roi_metrics(self, dataset: RoiPatchDataset) -> dict[str, list[RoiMetrics]]:
        """Collect ROI metrics for each group.

        Args:
            dataset: ROI patch dataset.

        Returns:
            Mapping of group name to list of ROI metrics.
        """

        grouped: dict[str, list[RoiMetrics]] = {}
        roi_scores = self._infer_roi_scores(dataset)

        for entry in dataset:
            image = load_rgb_image(entry.path)
            blob_stats = compute_blob_stats(self._segmentation_mask(image), self.infer_config.min_blob_area)
            mitosis_count = self._count_mitoses(image)
            roi_score = roi_scores.get(entry.path, 0.0)

            grouped.setdefault(entry.group, []).append(
                RoiMetrics(
                    blob_count=blob_stats.blob_count,
                    average_blob_area=blob_stats.average_area,
                    mitosis_count=mitosis_count,
                    roi_score=roi_score,
                )
            )

        return grouped

    def _infer_roi_scores(self, dataset: RoiPatchDataset) -> dict[Path, float]:
        """Infer ROI scores for patches using CNN_global.

        Args:
            dataset: ROI patch dataset.

        Returns:
            Mapping of patch path to ROI score.
        """

        if self.roi_scoring_model is None:
            self.logger.warning("CNN_global not provided. ROI scores set to 0.")
            return {}

        self.roi_scoring_model.eval()
        scores: dict[Path, float] = {}

        collate = _build_collate_fn(self.roi_transform)
        loader = DataLoader(
            dataset,
            batch_size=self.infer_config.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        with torch.no_grad():
            for batch in loader:
                images, paths = batch
                images = images.to(self.device)
                logits = self.roi_scoring_model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for path, pred in zip(paths, preds, strict=False):
                    scores[path] = float(pred + 1)

        return scores

    def _segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate segmentation mask for blob detection.

        Args:
            image: RGB image array.

        Returns:
            Binary mask for blob regions.
        """

        if not self.infer_config.use_segmentation or self.segmentation_model is None:
            return np.zeros(image.shape[:2], dtype=bool)

        self.segmentation_model.eval()
        tensor = _to_tensor(image).to(self.device)
        with torch.no_grad():
            output = self.segmentation_model(tensor.unsqueeze(0))
            logits = output.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(bool)
        return mask

    def _count_mitoses(self, image: np.ndarray) -> int:
        """Count mitoses using CNN_seg/CNN_det pipeline.

        Args:
            image: RGB patch image array.

        Returns:
            Estimated number of mitoses.
        """

        if not self.infer_config.use_segmentation or self.segmentation_model is None:
            return 0

        mask = self._segmentation_mask(image)
        labeled = measure.label(mask)
        regions = [region for region in measure.regionprops(labeled) if region.area >= self.infer_config.min_blob_area]
        if not self.infer_config.use_detection or self.detection_model is None:
            return len(regions)

        self.detection_model.eval()
        crops = [self._crop_detection_patch(image, region.centroid) for region in regions]
        if not crops:
            return 0

        crop_tensors = [cast(torch.Tensor, self.det_transform(crop)) for crop in crops]
        batch = torch.stack(crop_tensors).to(self.device)
        with torch.no_grad():
            logits = self.detection_model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return int(np.sum(preds == 1))

    def _crop_detection_patch(self, image: np.ndarray, centroid: tuple[float, float]) -> np.ndarray:
        """Crop a detection patch centered on a blob.

        Args:
            image: RGB image array.
            centroid: (row, col) centroid coordinates.

        Returns:
            Cropped RGB patch.
        """

        half = self.infer_config.detection_patch_size // 2
        row, col = int(centroid[0]), int(centroid[1])
        y1 = max(row - half, 0)
        y2 = min(row + half, image.shape[0])
        x1 = max(col - half, 0)
        x2 = min(col + half, image.shape[1])
        patch = image[y1:y2, x1:x2]
        return _pad_to_square(patch, self.infer_config.detection_patch_size)

    def _write_report(self, descriptors: dict[str, np.ndarray]) -> None:
        """Write descriptors to CSV.

        Args:
            descriptors: Mapping of group name to descriptor vector.
        """

        output_dir = Path(self.infer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "stage_four_hybrid_descriptors.csv"

        header = "group," + ",".join(f"feature_{idx}" for idx in range(1, 16)) + "\n"
        with report_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(header)
            for group, descriptor in descriptors.items():
                values = ",".join(f"{value:.6f}" for value in descriptor.tolist())
                file_handle.write(f"{group},{values}\n")

        self.logger.info("Saved hybrid descriptors to %s", report_path)


def _collect_roi_patches(root_dir: Path, extensions: tuple[str, ...]) -> list[RoiPatchEntry]:
    """Collect ROI patch files.

    Args:
        root_dir: Root directory containing ROI patch subfolders.
        extensions: Allowed image extensions.

    Returns:
        List of ROI patch entries.
    """

    entries: list[RoiPatchEntry] = []
    if not root_dir.exists():
        return entries

    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {ext.lower() for ext in extensions}:
            group = path.parent.name
            entries.append(RoiPatchEntry(path=path, group=group))
    return sorted(entries, key=lambda entry: entry.path.as_posix())


def _load_segmentation_model(
    config: HybridDescriptorInferenceConfig, device: torch.device
) -> MitosisSegmentationNet | None:
    """Load CNN_seg model if available."""

    if config.segmentation_model_path is None:
        return None
    model = MitosisSegmentationNet(pretrained=False)
    state = torch.load(config.segmentation_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def _load_detection_model(
    config: HybridDescriptorInferenceConfig, device: torch.device
) -> MitosisDetectionNet | None:
    """Load CNN_det model if available."""

    if config.detection_model_path is None:
        return None
    model = MitosisDetectionNet(pretrained=False)
    state = torch.load(config.detection_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def _load_roi_scoring_model(
    config: HybridDescriptorInferenceConfig, device: torch.device
) -> RoiScoringNet | None:
    """Load CNN_global model if available."""

    if config.roi_scoring_model_path is None:
        return None
    model = RoiScoringNet(pretrained=False)
    state = torch.load(config.roi_scoring_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def _build_imagenet_transform(size: int) -> transforms.Compose:
    """Build ImageNet-style preprocessing transform.

    Args:
        size: Resize target.

    Returns:
        Torchvision transform.
    """

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert RGB array to torch tensor without resize.

    Args:
        image: RGB image array.

    Returns:
        Normalized tensor.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return cast(torch.Tensor, transform(image))


def _pad_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """Pad a patch to the requested square size.

    Args:
        image: Patch image array.
        size: Target size.

    Returns:
        Square padded patch.
    """

    padded = np.zeros((size, size, 3), dtype=image.dtype)
    h, w = image.shape[:2]
    y_offset = max((size - h) // 2, 0)
    x_offset = max((size - w) // 2, 0)
    padded[y_offset : y_offset + h, x_offset : x_offset + w] = image
    return padded


def _build_collate_fn(
    transform: transforms.Compose,
) -> Callable[[Iterable[RoiPatchEntry]], tuple[torch.Tensor, list[Path]]]:
    """Build a collate function for ROI entries.

    Args:
        transform: Image transform to apply.

    Returns:
        Collate function.
    """

    def _collate_roi_entries(batch: Iterable[RoiPatchEntry]) -> tuple[torch.Tensor, list[Path]]:
        entries = list(batch)
        images = [load_rgb_image(entry.path) for entry in entries]
        tensors = torch.stack([cast(torch.Tensor, transform(image)) for image in images])
        paths = [entry.path for entry in entries]
        return tensors, paths

    return _collate_roi_entries
