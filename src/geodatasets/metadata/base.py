"""Base classes for metadata writers that save dataset-level and tile-level metadata to disk."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseMetadataWriter(ABC):
    """Abstract base class for metadata writers.

    Args:
        output_dir: Root directory where metadata files will be saved.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(self) -> None:
        """Save the recorded metadata to disk."""

    @abstractmethod
    def save_dataset_level(self) -> None:
        """Save any dataset-level metadata to disk."""
