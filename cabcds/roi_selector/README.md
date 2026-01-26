**Overview**

The ROI Selector module identifies and extracts candidate Regions of Interest (ROIs) from whole-slide images for downstream processing by the MF-CNN pipeline. It is implemented as a lightweight, configurable component that prepares ROI patches used for training and inference in the MF-CNN framework.

**Key Features**
- **Purpose:** Select tissue candidate patches likely to contain relevant pathology (mitoses, nuclei, etc.).
- **Integration:** Designed as a module within the MF-CNN toolchain; outputs match the expected input format for the MF-CNN networks.
- **Configurable:** Thresholds, patch size, and sampling strategies are configurable via the module config files.

**Implementation**
- The selector scans preprocessed slide regions and applies tissue/background heuristics and sampling rules to produce labeled ROI patches.
- Common operations include color-based filtering (e.g., blue-ratio), morphological filtering, and configurable positive/negative sampling used during training.
- Outputs are organized under the `output/roi_selector/` tree and can be consumed by the MF-CNN training and inference code.

**Usage (high level)**
- For training: run the sampling/training scripts to produce labeled ROI patches used by MF-CNN.
- For inference: use the ROI Selector to prepare candidate patches, then pass them to the MF-CNN inference pipeline.

**Files & Locations**
- Module code: `cabcds/train/roi_selector/` and helper utilities in `cabcds/train/roi_selector/utils/`.
- Outputs: `output/roi_selector/` (preprocessed patches and sampling outputs).

**Notes**
- See the project root README for full pipeline instructions and dependencies.
- Configuration parameters are defined in the module `config.py` files inside the ROI Selector folders.

If you want, I can expand this with example commands, configuration explanations, or a quick usage snippet.

