"""HDF5 dataset format for geodatasets."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from .base import BaseDataset


class HDF5Dataset(BaseDataset):
    """Dataset class for geodatasets in HDF5 format. Implements abstract BaseDataset."""

    def __init__(self, dataset_dir, tiles_csv, **kwargs) -> None:
        super().__init__(dataset_dir, tiles_csv, **kwargs)
        self._img_path = self.dataset_dir / "images" / "images.h5"
        self._mask_path = self.dataset_dir / "masks" / "masks.h5"
        self._img_h5: h5py.File | None = None
        self._mask_h5: h5py.File | None = None

        # Precompute once because bands never change after init
        sorted_positions = sorted(range(len(self.bands)), key=lambda i: self.bands[i])
        self._h5_bands = [self.bands[i] for i in sorted_positions]
        self._needs_reorder = self._h5_bands != self.bands

        self._inverse: list[int] | None = None
        if self._needs_reorder:
            inverse = [0] * len(sorted_positions)
            for new_pos, old_pos in enumerate(sorted_positions):
                inverse[old_pos] = new_pos
            self._inverse = inverse
        else:
            self._inverse = None

    def _load_tile(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_open()
        assert self._img_h5 is not None
        assert self._mask_h5 is not None

        key = f"{row['product_id']}_r{int(row['tile_grid_row'])}c{int(row['tile_grid_col'])}"
        img_chw = self._img_h5[key][self._h5_bands, ...]

        if self._needs_reorder:
            img_chw = img_chw[self._inverse, ...]

        mask = self._mask_h5[key][()]
        return img_chw.transpose(1, 2, 0), mask

    def _ensure_open(self) -> None:
        if self._img_h5 is None or not self._img_h5.id.valid:
            self._img_h5 = h5py.File(self._img_path, "r", swmr=True)
            self._mask_h5 = h5py.File(self._mask_path, "r", swmr=True)

    @property
    def _file(self) -> h5py.File | None:
        return self._img_h5

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_img_h5"] = None
        state["_mask_h5"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        if self._img_h5 is not None:
            self._img_h5.close()
        if self._mask_h5 is not None:
            self._mask_h5.close()
