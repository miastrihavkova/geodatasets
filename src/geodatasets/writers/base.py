"""Abstract base class for dataset writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseWriter(ABC):
    """Abstract base class for dataset writers. Defines the interface for writing to disk.

    Args:
        output_dir: Root directory where images/ and masks/ will be created.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write_tile(
        self,
        tile_id: str,
        image: np.ndarray,  # (H, W, 13) uint16
        mask: np.ndarray,  # (H, W) uint8
    ) -> tuple[str, str]:
        """Write one image tile and its corresponding mask to disk.

        Args:
            tile_id: Unique identifier for the tile.
            image: Image data to write.
            mask: Mask data to write.

        Returns:
            tuple[str, str]: Paths to the written image and mask files.
        """

    @abstractmethod
    def finalise(self) -> None:
        """Perform any necessary cleanup after all tiles have been written."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finalise()
