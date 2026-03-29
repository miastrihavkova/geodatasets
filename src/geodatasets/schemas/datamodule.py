"""Schema for datamodule configuration and dataset metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class S2DataModuleConfig(BaseModel):
    """Schema for configuration of the S2DataModule."""

    dataset_dir: Path
    tiles_csv: Path
    format: Literal["hdf5", "geotiff"]
    bands: list[int] | None = None
    batch_size: int = Field(default=16, gt=0)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True

    @field_validator("dataset_dir", "tiles_csv", mode="before")
    @classmethod
    def path_exists(cls, v: str | Path) -> Path:
        """Validates that the given path exists."""
        p = Path(v)
        if not p.exists():
            raise ValueError(f"path does not exist: {p}")
        return p

    @field_validator("bands")
    @classmethod
    def valid_band_indices(cls, v: list[int] | None) -> list[int] | None:
        """Validates that band indices are between 0 and 12 and are unique."""
        if v is None:
            return v
        for b in v:
            if not (0 <= b <= 12):
                raise ValueError(f"band index {b} out of range [0, 12]")
        if len(v) != len(set(v)):
            raise ValueError("band indices must be unique")
        return v
