"""Schemas for spectral band metadata and dataset-level metadata."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class BandInfoSchema(BaseModel):
    """Represents metadata about a single spectral band in the dataset."""

    band_id: int | str
    name: str
    center_wavelength_nm: int = Field(gt=0)
    bandwidth_nm: int = Field(gt=0)
    gsd_m: int = Field(gt=0)

    model_config = {"frozen": True}


class DatasetMetaSchema(BaseModel):
    """Represents dataset-level metadata, to be saved in dataset_metadata.json."""

    class_map: dict[str, str]
    bands: list[BandInfoSchema]

    @field_validator("class_map")
    @classmethod
    def keys_are_int_strings(cls, v: dict) -> dict:
        """Checks that all keys in class_map are strings that represent integers."""
        for k in v:
            if not k.isdigit():
                raise ValueError(f"class_map key must be an integer string, got '{k}'")
        return v

    @field_validator("bands")
    @classmethod
    def bands_not_empty(cls, v: list) -> list:
        """Validates that the bands list is not empty."""
        if not v:
            raise ValueError("bands list must not be empty")
        return v
