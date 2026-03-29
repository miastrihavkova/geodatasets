from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from .base import BaseDataset


class GeoTIFFDataset(BaseDataset):
    def _load_tile(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        img_path = self.dataset_dir / "images" / row["image_filename"]
        mask_path = self.dataset_dir / "masks" / row["mask_filename"]

        rasterio_bands = [b + 1 for b in self.bands]

        with rasterio.open(img_path) as src:
            img_chw = src.read(rasterio_bands)  # (C, H, W) uint16

        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W) uint8

        return img_chw.transpose(1, 2, 0), mask  # (H, W, C), (H, W)
