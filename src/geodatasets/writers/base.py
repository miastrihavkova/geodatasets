from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseWriter(ABC):
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
        pass

    @abstractmethod
    def finalise(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finalise()
