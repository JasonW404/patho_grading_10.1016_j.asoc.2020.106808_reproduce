"""Datasets and DataLoaders for MF-CNN training.

This module intentionally supports the on-disk layout present in this repo's
`dataset/` folder:

- TUPAC16 WSIs: `dataset/train/*.svs`, `dataset/test/*.svs`
- TUPAC16 proliferation labels: `dataset/train/ground_truth.csv` (row i => TR-i)
- Mitosis auxiliary dataset: image `.tif` tiles and centroid CSVs inside zips
  under `dataset/auxiliary_dataset_mitoses/`.

The paper describes centroid annotations that must be converted to pixel masks.
Here we follow the stated approach of using Blue-Ratio + Otsu to segment nuclei
blobs, then labeling the blob containing each centroid as positive.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from PIL import UnidentifiedImageError
from skimage import measure
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from cabcds.mf_cnn.config import MfcCnnConfig
from cabcds.roi_selector.utils.blue_ratio import compute_blue_ratio, otsu_mask


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalPatchIndexRow:
	"""One row from the CNN_global patch index CSV."""

	path: Path
	label: int
	slide_id: str


def read_global_patch_index(index_csv: Path) -> list[GlobalPatchIndexRow]:
	"""Read `index.csv` created by the MF-CNN global patch preparation step."""

	index_csv = Path(index_csv)
	rows: list[GlobalPatchIndexRow] = []
	with index_csv.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			path = Path(row["path"])
			label = int(row["label"])
			slide_id = str(row.get("slide_id") or path.parent.name)
			rows.append(GlobalPatchIndexRow(path=path, label=label, slide_id=slide_id))
	return rows


@dataclass(frozen=True)
class MitosisSampleKey:
	"""Unique key identifying a mitosis auxiliary image tile."""

	case_id: str
	tile_id: str

	@property
	def rel_image_path(self) -> str:
		return f"{self.case_id}/{self.tile_id}.tif"

	@property
	def rel_centroid_path(self) -> str:
		return f"mitoses_ground_truth/{self.case_id}/{self.tile_id}.csv"


def _read_rgb_from_zip(zip_path: Path, member: str) -> np.ndarray:
	with ZipFile(zip_path) as zf:
		with zf.open(member) as fp:
			image = Image.open(fp).convert("RGB")
			return np.asarray(image)


def _read_centroids_from_zip(zip_path: Path, member: str) -> list[tuple[int, int]]:
	with ZipFile(zip_path) as zf:
		with zf.open(member) as fp:
			content = fp.read().decode("utf-8", errors="replace")

	content = content.strip()
	if not content:
		return []

	centroids: list[tuple[int, int]] = []
	for line in content.splitlines():
		line = line.strip()
		if not line:
			continue
		parts = line.split(",")
		if len(parts) != 2:
			continue
		x = int(float(parts[0]))
		y = int(float(parts[1]))
		centroids.append((x, y))
	return centroids


def _read_rgb_from_path(image_path: Path) -> np.ndarray:
	with Image.open(image_path) as im:
		return np.asarray(im.convert("RGB"))


def _read_centroids_from_path(path: Path) -> list[tuple[int, int]]:
	"""Read centroid file from disk.

	We accept a simple CSV/TXT format: one "x,y" pair per line.
	Additional columns (e.g. probability) are ignored.
	"""
	content = Path(path).read_text(encoding="utf-8", errors="replace").strip()
	if not content:
		return []
	centroids: list[tuple[int, int]] = []
	for line in content.splitlines():
		line = line.strip()
		if not line:
			continue
		parts = line.split(",")
		if len(parts) < 2:
			continue
		x = int(float(parts[0]))
		y = int(float(parts[1]))
		centroids.append((x, y))
	return centroids


def _index_mitos14_structure(root: Path) -> list[tuple[Path, Path]]:
	"""Index specifically for MITOS14 directory structure.
    
    Structure:
      .../frames/x40/image_id.tiff
      .../mitosis/image_id_mitosis.csv
    """
	pairs: list[tuple[Path, Path]] = []
	# Search for all x40 tiff frames
	for image_path in root.rglob("frames/x40/*.tiff"):
		# Check for corresponding mitosis csv in sibling 'mitosis' folder
		# image_path: .../A03/frames/x40/A03_00Aa.tiff
		# expected csv: .../A03/mitosis/A03_00Aa_mitosis.csv
		
		# Go up 2 levels from image to reach 'A03' folder
		case_root = image_path.parent.parent.parent
		mitosis_dir = case_root / "mitosis"
		
		csv_name = f"{image_path.stem}_mitosis.csv"
		csv_path = mitosis_dir / csv_name
		
		if csv_path.exists():
			pairs.append((image_path, csv_path))
			
	return pairs


def _index_external_centroid_pairs(root: Path) -> list[tuple[Path, Path]]:
	"""Index an external mitosis dataset stored on disk.

	Expected (flexible) layouts that this function supports:
	- MITOS14 specific structure (detected automatically)
	- images anywhere under root (recursive)
	- centroid files stored as CSV/TXT with the same stem as the image
	  located either next to the image or under common annotation folders like
	  "annotations" / "annotation" / "centroids".

	If nothing is found, returns an empty list.
	"""
	root = Path(root)
	if not root.exists():
		return []

	# First, check for MITOS14 specific structure
	mitos14_pairs = _index_mitos14_structure(root)
	if mitos14_pairs:
		mitos14_pairs.sort(key=lambda x: str(x[0]))
		logger.info("Detected MITOS14 structure at %s: found %d pairs", str(root), len(mitos14_pairs))
		return mitos14_pairs

	def _zip_has_eocd(zip_path: Path) -> bool:
		"""Heuristically validate a ZIP by checking for the EOCD marker.

		Many interrupted downloads still start with a correct PK header, but are
		missing the end-of-central-directory record ("PK\x05\x06").
		"""
		sig = b"PK\x05\x06"
		try:
			size = zip_path.stat().st_size
			with zip_path.open("rb") as f:
				f.seek(max(0, size - 1024 * 1024))
				return f.read().rfind(sig) != -1
		except OSError:
			return False

	image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
	ann_dirs = [root / "annotations", root / "annotation", root / "centroids", root / "labels"]

	pairs: list[tuple[Path, Path]] = []
	found_any_image = False
	found_any_ann = False
	for image_path in root.rglob("*"):
		if not image_path.is_file():
			continue
		suffix = image_path.suffix.lower()
		if suffix in {".csv", ".txt"}:
			found_any_ann = True
		if suffix not in image_exts:
			continue

		found_any_image = True

		candidates = [
			image_path.with_suffix(".csv"),
			image_path.with_suffix(".txt"),
		]
		for ann_root in ann_dirs:
			candidates.append(ann_root / (image_path.stem + ".csv"))
			candidates.append(ann_root / (image_path.stem + ".txt"))

		centroid_path = next((p for p in candidates if p.exists() and p.is_file()), None)
		if centroid_path is None:
			continue
		pairs.append((image_path, centroid_path))

	pairs.sort(key=lambda x: str(x[0]))
	if not pairs:
		zip_files = list(root.glob("*.zip"))
		if zip_files and not found_any_image:
			bad = [p for p in zip_files if not _zip_has_eocd(p)]
			if bad:
				logger.warning(
					"External mitosis dataset root '%s' contains %d ZIP(s), but %d look truncated/corrupt (missing EOCD). "
					"Re-download them (with resume) before extracting.",
					str(root),
					len(zip_files),
					len(bad),
				)
			else:
				logger.warning(
					"External mitosis dataset root '%s' contains ZIP files but no extracted images were found. "
					"Extract the archives so images and centroid CSV/TXT files are on disk.",
					str(root),
				)
			return []
		if found_any_image and not found_any_ann:
			logger.warning(
				"External mitosis dataset root '%s' has images but no centroid CSV/TXT files were found; skipping.",
				str(root),
			)
		elif not found_any_image:
			logger.warning(
				"External mitosis dataset root '%s' has no supported image files; skipping.",
				str(root),
			)
	return pairs


def _centroids_to_mask_via_nuclei_blobs(
	image_rgb: np.ndarray,
	centroids_xy: Sequence[tuple[int, int]],
	*,
	min_area: int,
	fallback_radius: int,
) -> np.ndarray:
	"""Convert centroid points to a binary mask using BR+Otsu nuclei blobs.

	This follows the paper's intent: use BR+Otsu to segment nuclei blobs, then
	use pathologist-provided centroids to generate label images.

	Important: In practice, a centroid may fall outside the thresholded blob
	mask due to staining/threshold noise. To avoid silently dropping positives,
	we fall back to labeling a small disk around such centroids.

	Args:
		image_rgb: RGB image array (H, W, 3).
		centroids_xy: Sequence of (x, y) centroid coordinates.
		min_area: Minimum blob area to keep.
		fallback_radius: Radius (pixels) of a disk painted around centroids that
			do not fall inside any kept blob.

	Returns:
		Boolean mask of shape (H, W) where True indicates positive pixels.
	"""

	def _paint_disk(mask: np.ndarray, *, x: int, y: int, radius: int) -> None:
		if radius <= 0:
			if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
				mask[y, x] = True
			return

		y0 = max(0, y - radius)
		y1 = min(mask.shape[0], y + radius + 1)
		x0 = max(0, x - radius)
		x1 = min(mask.shape[1], x + radius + 1)
		h = y1 - y0
		w = x1 - x0
		if h <= 0 or w <= 0:
			return

		yy, xx = np.ogrid[:h, :w]
		yy = yy + y0 - y
		xx = xx + x0 - x
		disk = (xx * xx + yy * yy) <= (radius * radius)
		mask[y0:y1, x0:x1] |= disk

	blue = compute_blue_ratio(image_rgb)
	nuclei = otsu_mask(blue)
	labeled = measure.label(nuclei)

	positive_labels: set[int] = set()
	height, width = labeled.shape
	for x, y in centroids_xy:
		if 0 <= y < height and 0 <= x < width:
			label = int(labeled[y, x])
			if label > 0:
				positive_labels.add(label)

	positive_mask = np.zeros((height, width), dtype=bool)

	# Mark blobs that contain centroids (and are not tiny).
	kept_labels: set[int] = set()
	for region in measure.regionprops(labeled):
		if region.label in positive_labels and region.area >= int(min_area):
			positive_mask[labeled == region.label] = True
			kept_labels.add(region.label)

	# Fallback: if a centroid wasn't covered by a kept blob, paint a small disk.
	# This prevents dropping GT positives when BR+Otsu fails to capture a nucleus.
	for x, y in centroids_xy:
		if not (0 <= y < height and 0 <= x < width):
			continue
		label = int(labeled[y, x])
		if label <= 0 or label not in kept_labels:
			_paint_disk(positive_mask, x=int(x), y=int(y), radius=int(fallback_radius))

	return positive_mask


def _random_crop(
	image: np.ndarray,
	mask: np.ndarray,
	*,
	crop_size: int,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
	height, width = image.shape[:2]
	if height < crop_size or width < crop_size:
		raise ValueError(f"Image too small for crop: {(height, width)} < {crop_size}")

	top = int(rng.integers(0, height - crop_size + 1))
	left = int(rng.integers(0, width - crop_size + 1))
	image_crop = image[top : top + crop_size, left : left + crop_size]
	mask_crop = mask[top : top + crop_size, left : left + crop_size]
	return image_crop, mask_crop


def _to_tensor_rgb(image_rgb: np.ndarray) -> Tensor:
	image_rgb = np.ascontiguousarray(image_rgb)
	if not image_rgb.flags.writeable:
		image_rgb = image_rgb.copy()
	return torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float() / 255.0


def _resize_tensor_image(tensor_chw: Tensor, size: int) -> Tensor:
	return resize(tensor_chw, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)


def _imagenet_normalize(tensor_chw: Tensor) -> Tensor:
	"""Normalize an image tensor for ImageNet-pretrained backbones.

	Torchvision pretrained models expect inputs normalized with:
	mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
	"""

	return normalize(
		tensor_chw,
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225],
	)


class MitosisSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
	"""Random-crop dataset for training `CNNSeg`.

	Each sample is a random crop from a random mitosis auxiliary tile.
	Target is a 2-class mask (0=background, 1=mitosis blob).
	"""

	def __init__(
		self,
		*,
		image_zip_parts: Sequence[Path],
		ground_truth_zip: Path,
		external_roots: Sequence[Path] | None = None,
		crop_size: int = 512,
		nuclei_min_area: int = 10,
		centroid_fallback_radius: int = 6,
		length: int = 10_000,
		seed: int = 1337,
		augment: bool = True,
	) -> None:
		self.image_zip_parts = tuple(image_zip_parts)
		self.ground_truth_zip = Path(ground_truth_zip)
		self.external_roots = tuple(Path(p) for p in (external_roots or ()))
		self.crop_size = int(crop_size)
		self.nuclei_min_area = int(nuclei_min_area)
		self.centroid_fallback_radius = int(centroid_fallback_radius)
		self.length = int(length)
		self.seed = int(seed)
		self.augment = bool(augment)

		self._keys = _index_mitosis_tiles(self.image_zip_parts, ground_truth_zip=self.ground_truth_zip)
		self._external_pairs: list[tuple[Path, Path]] = []
		for root in self.external_roots:
			self._external_pairs.extend(_index_external_centroid_pairs(root))
		if self._external_pairs:
			logger.info("External centroid tiles indexed for CNN_seg: %d", len(self._external_pairs))
		if not self._keys:
			raise ValueError("No mitosis tiles found in provided zip parts")

	def __len__(self) -> int:  # noqa: D401
		return self.length

	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		rng = np.random.default_rng(self.seed + int(index))
		total = len(self._keys) + len(self._external_pairs)
		pick = int(rng.integers(0, total)) if total > 0 else 0
		if pick < len(self._keys):
			key = self._keys[pick]
			zip_path = _find_tile_zip(self.image_zip_parts, key.rel_image_path)
			image = _read_rgb_from_zip(zip_path, key.rel_image_path)
			centroids = _read_centroids_from_zip(self.ground_truth_zip, key.rel_centroid_path)
		else:
			image_path, centroid_path = self._external_pairs[pick - len(self._keys)]
			image = _read_rgb_from_path(image_path)
			centroids = _read_centroids_from_path(centroid_path)
		mask = _centroids_to_mask_via_nuclei_blobs(
			image,
			centroids,
			min_area=self.nuclei_min_area,
			fallback_radius=self.centroid_fallback_radius,
		)


		image_crop, mask_crop = _random_crop(image, mask, crop_size=self.crop_size, rng=rng)

		if self.augment:
			if float(rng.random()) < 0.5:
				image_crop = image_crop[:, ::-1].copy()
				mask_crop = mask_crop[:, ::-1].copy()
			if float(rng.random()) < 0.5:
				image_crop = image_crop[::-1, :].copy()
				mask_crop = mask_crop[::-1, :].copy()

		image_tensor = _to_tensor_rgb(image_crop)
		mask_tensor = torch.from_numpy(mask_crop.astype(np.int64))
		return image_tensor, mask_tensor


class MitosisDetectionDataset(Dataset[tuple[Tensor, Tensor]]):
	"""Candidate crop dataset for training `CNNDet`.

	Candidates are nuclei blobs (BR+Otsu). A blob is positive if it is the
	blob containing a mitosis centroid.
	"""

	def __init__(
		self,
		*,
		image_zip_parts: Sequence[Path],
		ground_truth_zip: Path,
		crop_size: int = 80,
		output_size: int = 227,
		nuclei_min_area: int = 10,
		length: int = 20_000,
		seed: int = 1337,
		augment: bool = True,
	) -> None:
		self.image_zip_parts = tuple(image_zip_parts)
		self.ground_truth_zip = Path(ground_truth_zip)
		self.crop_size = int(crop_size)
		self.output_size = int(output_size)
		self.nuclei_min_area = int(nuclei_min_area)
		self.length = int(length)
		self.seed = int(seed)
		self.augment = bool(augment)

		self._keys = _index_mitosis_tiles(self.image_zip_parts, ground_truth_zip=self.ground_truth_zip)
		if not self._keys:
			raise ValueError("No mitosis tiles found in provided zip parts")

	def __len__(self) -> int:  # noqa: D401
		return self.length

	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		rng = np.random.default_rng(self.seed + 10_000 + int(index))
		key = self._keys[int(rng.integers(0, len(self._keys)))]
		zip_path = _find_tile_zip(self.image_zip_parts, key.rel_image_path)

		image = _read_rgb_from_zip(zip_path, key.rel_image_path)
		centroids = _read_centroids_from_zip(self.ground_truth_zip, key.rel_centroid_path)

		blue = compute_blue_ratio(image)
		nuclei = otsu_mask(blue)
		labeled = measure.label(nuclei)
		height, width = labeled.shape

		positive_labels: set[int] = set()
		for x, y in centroids:
			if 0 <= y < height and 0 <= x < width:
				label = int(labeled[y, x])
				if label > 0:
					positive_labels.add(label)

		regions = [r for r in measure.regionprops(labeled) if r.area >= self.nuclei_min_area]
		if not regions:
			# Fallback: return a random crop labeled negative.
			dummy_mask = np.zeros((height, width), dtype=bool)
			image_crop, _ = _random_crop(image, dummy_mask, crop_size=max(self.crop_size, 64), rng=rng)
			image_tensor = _resize_tensor_image(_to_tensor_rgb(image_crop), self.output_size)
			return image_tensor, torch.tensor(0, dtype=torch.long)

		region = regions[int(rng.integers(0, len(regions)))]
		cy, cx = region.centroid  # (row, col)
		cy_i, cx_i = int(cy), int(cx)

		half = self.crop_size // 2
		top = max(0, min(height - self.crop_size, cy_i - half))
		left = max(0, min(width - self.crop_size, cx_i - half))
		patch = image[top : top + self.crop_size, left : left + self.crop_size]

		if self.augment and float(rng.random()) < 0.5:
			patch = patch[:, ::-1].copy()

		label = 1 if int(labeled[cy_i, cx_i]) in positive_labels else 0
		image_tensor = _resize_tensor_image(_to_tensor_rgb(patch), self.output_size)
		return image_tensor, torch.tensor(label, dtype=torch.long)


@dataclass(frozen=True)
class DetPatchIndexRow:
	"""One row from a prepared CNN_det patch index CSV."""

	path: Path
	label: int
	case_id: str
	tile_id: str
	cx: int
	cy: int
	is_gt_forced: int = 0


def read_det_patch_index(index_csv: Path) -> list[DetPatchIndexRow]:
	"""Read `index.csv` produced by CNN_det candidate preparation."""

	rows: list[DetPatchIndexRow] = []
	with Path(index_csv).open("r", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			is_gt_forced = int(float(row.get("is_gt_forced") or 0))
			rows.append(
				DetPatchIndexRow(
					path=Path(row["path"]),
					label=int(row["label"]),
					case_id=str(row.get("case_id") or ""),
					tile_id=str(row.get("tile_id") or ""),
					cx=int(float(row.get("cx") or 0)),
					cy=int(float(row.get("cy") or 0)),
					is_gt_forced=int(is_gt_forced),
				)
			)
	return rows


class PreparedMitosisDetectionPatchDataset(Dataset[tuple[Tensor, Tensor]]):
	"""Disk-backed dataset for CNN_det training.

	Expects 80x80 RGB patches saved as PNG/JPG and indexed by `index.csv`.
	The model still consumes 227x227 tensors, so samples are resized on load.
	"""

	def __init__(
		self,
		*,
		index_csv: Path | None = None,
		rows: list[DetPatchIndexRow] | None = None,
		output_size: int = 227,
		normalize_imagenet: bool = True,
	) -> None:
		self.index_csv = Path(index_csv) if index_csv else None
		self.output_size = int(output_size)
		self.normalize_imagenet = bool(normalize_imagenet)
		
		if rows is not None:
			self._rows = rows
		elif self.index_csv is not None:
			self._rows = read_det_patch_index(self.index_csv)
		else:
			raise ValueError("Either index_csv or rows must be provided")

		if not self._rows:
			# warn but allow empty for edge cases? No, dataset should have data.
			# But if a split is empty (e.g. test set too small), it might crash.
			# Keeping original behavior of raising error for now, but maybe relax later.
			raise ValueError(f"Empty CNN_det index.")

	def __len__(self) -> int:  # noqa: D401
		return len(self._rows)

	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		row = self._rows[int(index)]
		with Image.open(row.path) as im:
			image = im.convert("RGB")
			tensor = _to_tensor_rgb(np.asarray(image))
		tensor = _resize_tensor_image(tensor, self.output_size)
		if self.normalize_imagenet:
			tensor = _imagenet_normalize(tensor)
		label = int(row.label)
		return tensor, torch.tensor(label, dtype=torch.long)


class GlobalScoringPatchDataset(Dataset[tuple[Tensor, Tensor]]):
	"""Patch dataset for training `CNNGlobal`.

	Expects patch files on disk plus a slide-level score mapping.
	"""

	def __init__(
		self,
		*,
		patch_index_csv: Path | None = None,
		rows: Sequence[GlobalPatchIndexRow] | None = None,
		transform: object | None = None,
		output_size: int = 227,
		normalize_imagenet: bool = True,
	) -> None:
		if patch_index_csv is None and rows is None:
			raise ValueError("Either patch_index_csv or rows must be provided")

		self.patch_index_csv = Path(patch_index_csv) if patch_index_csv is not None else None
		self.output_size = int(output_size)
		self.transform = transform
		self.normalize_imagenet = bool(normalize_imagenet)

		self._rows: list[tuple[Path, int]] = []
		if rows is not None:
			for r in rows:
				self._rows.append((Path(r.path), int(r.label)))
		else:
			assert self.patch_index_csv is not None
			with self.patch_index_csv.open("r", newline="") as f:
				reader = csv.DictReader(f)
				for row in reader:
					path = Path(row["path"])
					label = int(row["label"])  # expected 1..3 or 0..2
					self._rows.append((path, label))

		if not self._rows:
			raise ValueError(
				f"No rows found in patch index: {self.patch_index_csv}" if self.patch_index_csv else "No rows provided"
			)

	def __len__(self) -> int:  # noqa: D401
		return len(self._rows)

	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		# Some extracted patches may be corrupted if an extraction process was
		# interrupted mid-write. Be defensive and retry with another sample.
		base_index = int(index)
		last_err: Exception | None = None
		for attempt in range(5):
			path, label = self._rows[(base_index + attempt) % len(self._rows)]
			try:
				with Image.open(path) as im:
					image = im.convert("RGB")
					if self.transform is not None:
						transformed = self.transform(image)
						# We expect PIL-in/PIL-out transforms here (augmentation). If a
						# tensor is returned, we still accept it.
						if isinstance(transformed, torch.Tensor):
							image_tensor = transformed
						else:
							image_tensor = _to_tensor_rgb(np.asarray(transformed))
					else:
						image_tensor = _to_tensor_rgb(np.asarray(image))
				break
			except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
				last_err = e
				logger.warning("Bad global patch file; skipping: %s (%s)", str(path), type(e).__name__)
		else:
			raise RuntimeError(f"Failed to load any valid global patch near index={base_index}") from last_err

		# Ensure canonical size for AlexNet, then normalize.
		image_tensor = _resize_tensor_image(image_tensor, self.output_size)
		if self.normalize_imagenet:
			image_tensor = _imagenet_normalize(image_tensor)
		# Normalize labels to 0..2 for CrossEntropyLoss if they are 1..3.
		if label in (1, 2, 3):
			label = label - 1
		return image_tensor, torch.tensor(label, dtype=torch.long)


def _index_mitosis_tiles(
	zip_parts: Sequence[Path],
	*,
	ground_truth_zip: Path | None = None,
) -> list[MitosisSampleKey]:
	"""Build an index of available `.tif` tile keys across zip parts.

	The auxiliary mitosis dataset ships image tiles and centroid CSVs in separate
	archives. In practice, some tiles may be missing their corresponding
	centroid CSVs. If `ground_truth_zip` is provided, this function filters the
	keys to only those that have a matching centroid member.
	"""

	keys: list[MitosisSampleKey] = []
	seen: set[str] = set()
	for zip_path in zip_parts:
		with ZipFile(zip_path) as zf:
			for name in zf.namelist():
				if not name.lower().endswith(".tif"):
					continue
				# Format is like "01/01.tif"
				try:
					case_id, filename = name.split("/", 1)
				except ValueError:
					continue
				tile_id = Path(filename).stem
				key = MitosisSampleKey(case_id=case_id, tile_id=tile_id)
				if key.rel_image_path in seen:
					continue
				seen.add(key.rel_image_path)
				keys.append(key)

	keys.sort(key=lambda k: (k.case_id, k.tile_id))
	if ground_truth_zip is None:
		return keys

	ground_truth_zip = Path(ground_truth_zip)
	with ZipFile(ground_truth_zip) as zf:
		available = set(zf.namelist())

	filtered = [key for key in keys if key.rel_centroid_path in available]
	missing = len(keys) - len(filtered)
	if missing > 0:
		logger.warning(
			"Mitosis auxiliary dataset: %d/%d tiles are missing centroid CSVs in %s; filtering them out",
			missing,
			len(keys),
			str(ground_truth_zip),
		)
	return filtered


def _find_tile_zip(zip_parts: Sequence[Path], rel_member: str) -> Path:
	for zip_path in zip_parts:
		with ZipFile(zip_path) as zf:
			if rel_member in zf.namelist():
				return zip_path
	raise FileNotFoundError(f"Tile member not found in any zip part: {rel_member}")


def load_tupac_train_scores(ground_truth_csv: Path) -> dict[str, int]:
	"""Load TUPAC train slide scores (1..3) from `ground_truth.csv`.

	The file in this repo contains two comma-separated numeric columns without a
	header. Per TUPAC convention, row i corresponds to slide `TUPAC-TR-{i:03d}`.
	Only the first column (integer score 1/2/3) is used for `CNN_global`.
	"""

	scores: dict[str, int] = {}
	with Path(ground_truth_csv).open("r", encoding="utf-8") as f:
		for idx, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			parts = line.split(",")
			score = int(float(parts[0]))
			slide_id = f"TUPAC-TR-{idx:03d}"
			scores[slide_id] = score
	return scores


def build_default_mf_cnn_train_loaders(
	config: MfcCnnConfig,
	*,
	batch_size: int = 4,
	num_workers: int = 0,
	seg_length: int = 10_000,
	det_length: int = 20_000,
	seg_augment: bool = True,
	det_augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
	"""Build segmentation and detection loaders from the mitosis auxiliary zips.

	Global scoring patches are typically prepared separately (WSI patch extraction
	is expensive), so this helper only returns (seg, det) loaders.
	"""

	external_roots: list[Path] = []
	potential = [Path(config.mitos12_dir), Path(config.mitos14_dir)]
	existing = [p for p in potential if p.exists()]
	if existing and not config.use_external_mitosis_datasets:
		logger.info(
			"Found external dataset directories (%s) but `use_external_mitosis_datasets` is false; continuing without them.",
			", ".join(str(p) for p in existing),
		)
	elif config.use_external_mitosis_datasets:
		# Allow users to enable any subset of supported external datasets.
		external_roots = existing
		if not external_roots:
			raise FileNotFoundError(
				"Requested external mitosis datasets, but none of these paths exist: "
				+ ", ".join(str(p) for p in potential)
			)
		missing = [p for p in potential if not p.exists()]
		if missing:
			logger.warning(
				"Some external dataset paths do not exist and will be skipped: %s",
				", ".join(str(p) for p in missing),
			)

	base = config.tupac_aux_mitoses_dir
	image_zips = [base / rel for rel in config.mitoses_image_zip_parts]
	ground_truth_zip = base / config.mitoses_ground_truth_zip

	seg_ds = MitosisSegmentationDataset(
		image_zip_parts=image_zips,
		ground_truth_zip=ground_truth_zip,
		external_roots=external_roots,
		crop_size=config.seg_crop_size,
		nuclei_min_area=config.nuclei_min_area,
		centroid_fallback_radius=config.centroid_fallback_radius,
		length=seg_length,
		seed=config.seed,
		augment=bool(seg_augment),
	)
	det_ds = MitosisDetectionDataset(
		image_zip_parts=image_zips,
		ground_truth_zip=ground_truth_zip,
		crop_size=config.det_crop_size,
		output_size=config.alexnet_input_size,
		nuclei_min_area=config.nuclei_min_area,
		length=det_length,
		seed=config.seed,
		augment=bool(det_augment),
	)

	seg_loader = DataLoader(seg_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
	det_loader = DataLoader(det_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
	return seg_loader, det_loader

