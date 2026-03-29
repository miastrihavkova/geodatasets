from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .base import BaseWriter


class HDF5Writer(BaseWriter):
    def __init__(self, output_dir: Path, chunk_hw: int = 256) -> None:
        super().__init__(output_dir)
        self._chunk_hw = chunk_hw
        self._img_file = h5py.File(self.images_dir / "images.h5", "w")
        self._mask_file = h5py.File(self.masks_dir / "masks.h5", "w")

    def write_tile(self, tile_id: str, image: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
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
        self._img_file.close()
        self._mask_file.close()
