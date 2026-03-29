"""LightningDataModule for Sentinel-2 datasets."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from geodatasets.datasets.geotiff import GeoTIFFDataset
from geodatasets.datasets.hdf5 import HDF5Dataset

FORMAT_MAP = {"geotiff": GeoTIFFDataset, "hdf5": HDF5Dataset}


class S2DataModule(pl.LightningDataModule):
    """Provides Sentinel-2 dataloaders for training and evaluation.

    Args:
        dataset_dir: Location of the dataset, containing the image and mask files.
        tiles_csv: Path to the CSV file describing the tiles and their metadata.
        format: The format in which the dataset is stored, either 'geotiff' or 'hdf5'.
        bands: List of band indices to load (0-12), or None to load all bands. Defaults to None.
        batch_size: Batch size for the dataloaders. Defaults to 16.
        num_workers: Number of worker processes for data loading. Defaults to 4.
        pin_memory: Whether to use pinned memory for data loading. Defaults to True.
        train_transform: Optional transform to apply to training images and masks.
        val_transform: Optional transform to apply to validation and test images and masks.

    Raises:
        ValueError: If an unknown format is specified, or if indices are out of range or not unique.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        tiles_csv: str | Path,
        format: str = "geotiff",
        bands: list[int] | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform=None,
        val_transform=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["train_transform", "val_transform"])

        if format not in FORMAT_MAP:
            raise ValueError(f"unknown format '{format}', choose from {list(FORMAT_MAP)}")

        self.dataset_dir = Path(dataset_dir)
        self.tiles_csv = Path(tiles_csv)
        self.format = format
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transform = train_transform
        self.val_transform = val_transform

        self._dataset_cls = FORMAT_MAP[format]
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    @classmethod
    def from_yaml(cls, config_path: str | Path, **overrides) -> S2DataModule:
        """Loads config, if provided in YAML format.

        Args:
            config_path: Path to the YAML config file.
            **overrides: Any config values to override from the YAML file.

        Returns:
            S2DataModule: An instance of S2DataModule initialized with the config values.
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg.update(overrides)
        return cls(**cfg)

    def _make_dataset(self, split: str, transform=None):
        return self._dataset_cls(
            dataset_dir=self.dataset_dir,
            tiles_csv=self.tiles_csv,
            split=split,
            bands=self.bands,
            transform=transform,
        )

    def setup(self, stage: str | None = None) -> None:
        """Sets up the datasets for the specified stage."""
        if stage in ("fit", None):
            self._train_ds = self._make_dataset("train", self.train_transform)
            self._val_ds = self._make_dataset("val", self.val_transform)
        if stage in ("test", None):
            self._test_ds = self._make_dataset("test", self.val_transform)

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Provides the training dataloader."""
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Provides the validation dataloader."""
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Provides the test dataloader."""
        return self._make_loader(self._test_ds, shuffle=False)
