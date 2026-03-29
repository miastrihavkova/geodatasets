from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseMetadataWriter(ABC):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def save_dataset_level(self) -> None:
        pass
