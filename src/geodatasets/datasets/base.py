from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VALID_SPLITS = {"train", "val", "test"}


class BaseDataset(ABC, Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        tiles_csv: Path,
        split: str | None = None,
        bands: Sequence[int] | None = None,
        transform=None,
        target_transform=None,
    ) -> None:

        self.dataset_dir = Path(dataset_dir)
        self.bands = list(bands) if bands is not None else list(range(13))
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(tiles_csv)

        if split is not None:
            if split not in VALID_SPLITS:
                raise ValueError(f"invalid split '{split}', choose from {VALID_SPLITS}")
            df = df[df["split"] == split].reset_index(drop=True)

        self.tiles = df

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")

        row = self.tiles.iloc[idx]
        image, mask = self._load_tile(row)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.target_transform:
            mask_tensor = self.target_transform(mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "tile_id": row["tile_id"],
            "product_id": row["product_id"],
            "cloud_percent": row.get("cloud_percent", float("nan")),
        }

    @abstractmethod
    def _load_tile(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Returns image (H, W, C) uint16 and mask (H, W) uint8."""
