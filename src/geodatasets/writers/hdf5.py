"""Writer for HDF5 format using h5py."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .base import BaseWriter


class HDF5Writer(BaseWriter):
    """Writes one HDF5 file per tile with gzip compression.

    Args:
        output_dir: Root directory where images/ and masks/ will be created.
        chunk_hw: Chunk height and width in pixels. Defaults to 256.
    """

    def __init__(self, output_dir: Path, chunk_hw: int = 256) -> None:
        super().__init__(output_dir)
        self._chunk_hw = chunk_hw
        self._img_file = h5py.File(self.images_dir / "images.h5", "w")
        self._mask_file = h5py.File(self.masks_dir / "masks.h5", "w")

    def write_tile(self, tile_id: str, image: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
        """Write one image tile and its corresponding mask into the HDF5 files.

        Args:
            tile_id: Unique identifier used as the dataset key within the HDF5 file.
            image: Image array of shape (H, W, C) with dtype uint16.
            mask: Mask array of shape (H, W) with dtype uint8.

        Returns:
            A tuple of (image_filename, mask_filename), both pointing to the
            shared HDF5 files rather than individual tile files.
        """
        H, W, C = image.shape
        img_chw = image.transpose(2, 0, 1)

        self._img_file.create_dataset(
            tile_id,
            data=img_chw,
            dtype="uint16",
            chunks=(1, self._chunk_hw, self._chunk_hw),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        self._mask_file.create_dataset(
            tile_id,
            data=mask,
            dtype="uint8",
            chunks=(self._chunk_hw, self._chunk_hw),
            compression="gzip",
            compression_opts=4,
        )

        return "images.h5", "masks.h5"

    def finalise(self) -> None:
        """Close both HDF5 files and flush all pending writes to disk."""
        self._img_file.close()
        self._mask_file.close()
