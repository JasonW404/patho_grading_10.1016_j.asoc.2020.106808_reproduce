"""Project entry point for CABCDS reproduction tasks."""

from __future__ import annotations

from cabcds.data_preparation.config import load_stage_one_config
from cabcds.data_preparation.pipeline import StageOnePreprocessor


def main() -> None:
	"""Main entry point."""

	config = load_stage_one_config()
	preprocessor = StageOnePreprocessor(config)
	preprocessor.run()


if __name__ == "__main__":
	main()
